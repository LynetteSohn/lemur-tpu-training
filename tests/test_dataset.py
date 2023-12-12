import json

import mlxu

from xtpu.data import DatasetFactory
from xtpu.model import LLaMAConfig

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
)


def main(argv):
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)

    for _batch, dataset_metrics in dataset:
        # print(batch)
        print(json.dumps(dataset_metrics, indent=2))
        # print(batch["input_tokens"].shape)
        break


if __name__ == "__main__":
    mlxu.run(main)
