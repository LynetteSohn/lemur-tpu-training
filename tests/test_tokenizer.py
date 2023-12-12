import math

import flax
import jax.numpy as jnp
import mlxu
from flax.core.frozen_dict import freeze, unfreeze

from xtpu.checkpoint import StreamingCheckpointer
from xtpu.model import LLaMAConfig

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint="params::/.../models/llama-7b-tpu/weight.model",
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    tokenizer=LLaMAConfig.get_tokenizer_config(),
)


def main(argv):

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        "",
        enable=True,
    )
    train_state, restored_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint)
    restored_params = unfreeze(restored_params)
    num_new_tokens = 19

    input_embeddings = restored_params["params"]["transformer"]["wte"]["embedding"]
    output_embeddings = restored_params["params"]["lm_head"]["kernel"]

    num_new_token_embeddings = (
        8 * math.ceil((input_embeddings.shape[0] + num_new_tokens) / 8.0) - input_embeddings.shape[0]
    )

    input_embeddings_avg = jnp.mean(input_embeddings, axis=0)
    output_embeddings_avg = jnp.mean(output_embeddings, axis=1)

    new_input_embeddings = jnp.concatenate(
        (input_embeddings, jnp.repeat(input_embeddings_avg[None, :], num_new_token_embeddings, axis=0)), axis=0
    )
    new_output_embeddings = jnp.concatenate(
        (output_embeddings, jnp.repeat(output_embeddings_avg[:, None], num_new_token_embeddings, axis=1)), axis=1
    )

    print(f"New input embeddings shape: {str(new_input_embeddings.shape)}")
    print(f"New output embeddings shape: {str(new_output_embeddings.shape)}")

    # print("Number of new tokens: ", num_new_tokens)


if __name__ == "__main__":
    mlxu.run(main)
