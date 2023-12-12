import math
import pprint
from functools import partial

import jax
import jax.numpy as jnp
import mlxu
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS  # noqa: N817
from tqdm import tqdm, trange

from xtpu.checkpoint import StreamingCheckpointer
from xtpu.data import DatasetFactory
from xtpu.jax_utils import (
    JaxDistributedConfig,
    JaxRNG,
    average_metrics,
    cross_entropy_loss_and_accuracy,
    get_float_dtype_by_name,
    get_weight_decay_mask,
    global_norm,
    make_shard_and_gather_fns,
    match_partition_rules,
    next_rng,
    set_random_seed,
    with_sharding_constraint,
)
from xtpu.model import FlaxLLaMAForCausalLMModule, LLaMAConfig
from xtpu.optimizers import OptimizerFactory
# from jax import config
# config.update("jax_debug_nans", True)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim="1,-1,1",
    dtype="fp32",
    total_steps=10000,
    load_llama_config="",
    update_llama_config="",
    load_checkpoint="",
    load_dataset_state="",
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    pprint.pprint(flags_config_dict)
    pprint.pprint(f"JAX process index: {jax.process_index()}")
    pprint.pprint(f"JAX process count: {jax.process_count()}")
    pprint.pprint(f"JAX host count: {jax.host_count()}")
    pprint.pprint(f"JAX host id: {jax.host_id()}")
    pprint.pprint(f"JAX devices: {jax.devices()}")
    pprint.pprint(f"JAX device count: {jax.device_count()}")
    pprint.pprint(f"JAX local devices: {jax.local_devices()}")
    pprint.pprint(f"JAX local device count: {jax.local_device_count()}")
    pprint.pprint(f"Loading tokenizer: {FLAGS.tokenizer}")
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    special_tokens_dict = {
        "additional_special_tokens": [
            "<|endoftext|>",
            "<fim_prefix>",
            "<fim_middle>",
            "<fim_suffix>",
            "<fim_pad>",
            "<reponame>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<empty_output>",
            "<commit_before>",
            "<commit_msg>",
            "<commit_after>",
        ]
    }

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    pprint.pprint(f"Added {num_new_tokens} new tokens")
    pprint.pprint(f"Loading dataset: {FLAGS.train_dataset}")
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != "":
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(FLAGS.eval_dataset, dataset.tokenizer)
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != "":
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != "":
        llama_config.update(dict(eval(FLAGS.update_llama_config)))  # noqa: PGH001

    llama_config.update(
        {
            "bos_token_id": dataset.tokenizer.bos_token_id,
            "eos_token_id": dataset.tokenizer.eos_token_id,
        }
    )
    num_new_token_embeddings = 0
    if llama_config.vocab_size < dataset.vocab_size:
        pprint.pprint(
            f"Vocab size mismatch between dataset ({dataset.vocab_size}) and config ({llama_config.vocab_size})"
        )
        pprint.pprint(f"Using dataset vocab size: {dataset.vocab_size}")

        pprint.pprint(f"Mesh dim: {FLAGS.mesh_dim}")
        multiple_factor = max(8, int(FLAGS.mesh_dim.split(",")[-1]))
        pprint.pprint(f"Multiple factor: {multiple_factor}")
        if dataset.vocab_size % multiple_factor != 0:
            pprint.pprint(f"Making the vocab size multiple of {multiple_factor}")
            orginal_model_vocab_size = llama_config.vocab_size
            llama_config.update({"vocab_size": multiple_factor * math.ceil(dataset.vocab_size / multiple_factor)})
            pprint.pprint(f"New vocab size: {llama_config.vocab_size}")
            num_new_token_embeddings = llama_config.vocab_size - orginal_model_vocab_size
        else:
            llama_config.update({"vocab_size": dataset.vocab_size})
            num_new_token_embeddings = llama_config.vocab_size - orginal_model_vocab_size

        pprint.pprint(f"Num of new token embeddings: {num_new_token_embeddings}")
    pprint.pprint(f"Llama config: {llama_config.to_dict()}")
    model = FlaxLLaMAForCausalLMModule(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype))

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer, get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))

        def loss_and_accuracy(params):
            logits = model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": optimizer_info["learning_rate_schedule"](train_state.step),
            "gradient_norm": global_norm(grads),
            "param_norm": global_norm(train_state.params),
        }
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            train_state.params,
            batch["input_tokens"],
            deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
        metrics = {
            "eval_loss": loss,
            "eval_accuracy": accuracy,
        }
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(LLaMAConfig.get_partition_rules(), train_state_shapes)

    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(init_fn, in_shardings=PS(), out_shardings=train_state_partition)

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = {
            "step": step,
            "variant": variant,
            "flags": flags_config_dict,
            "llama_config": llama_config.to_dict(),
        }
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != "":
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(train_state, sharded_rng, batch)

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(train_state, sharded_rng, eval_batch)
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)