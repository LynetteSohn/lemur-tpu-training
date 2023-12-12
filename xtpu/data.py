import dataclasses
import json
import os
import pprint
import random
import time
from functools import partial
from multiprocessing import Pool

import h5py
import mlxu
import numpy as np
from datasets import load_dataset
from google.cloud import storage
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from tqdm import tqdm, trange


class DatasetFactory:
    """Datset builder class."""

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = "huggingface"
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        config.multisource_json_dataset = MultiSourceJsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == "huggingface":
            return HuggingfaceDataset(config.huggingface_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == "json":
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == "multisource_json":
            return MultiSourceJsonDataset(config.multisource_json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")

    def __init__(self):
        raise ValueError("DatasetFactory is a static class and should not be instantiated.")


class TextProcessor:
    """Example processor that converts a dictionary of texts into tokens."""

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ""
        config.fields = ""
        config.subfield_separator = " "
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ""
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert (
            self.config.fields != "" or self.config.fields_from_example != ""
        ), "Either fields or fields_from_example must be specified."
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = ()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != "":
            fields = example[self.config.fields_from_example].split(",")
        else:
            fields = self.config.fields.split(",")

        for i, field in enumerate(fields):
            if field.startswith("[") and field.endswith("]"):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == "<|bos|>":
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == "<|eos|>":
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split("+")
                text = self.config.subfield_separator.join([example[subfield] for subfield in subfields])
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset:
    """Huggingface dataset, where the dataset is loaded using the huggingface
    datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = "c4"
        config.name = "en"
        config.split = "train"
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != "" else None
        split = self.config.split if self.config.split != "" else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(self.config.path, name, split=split, streaming=self.config.streaming).shuffle()

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        "dataset_example_index": index,
                        "dataset_total_tokens": total_tokens,
                    }
                    batch = {
                        "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch["input_tokens"][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return {"config": self.config}

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            self.config.update(ConfigDict(state_dict["config"]))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset:
    """JSON dataset, where each line of the data file contains a JSON
    dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ""
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start
        # self.storage_client = storage.Client()
        # self.bucket = self.storage_client.get_bucket("sfr-tpu-us-central2-research")

    def parse_json(self, line):
        if not line or line == "\n":
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f"Error parsing json line:\n{line}")
            return None
        return data

    # def json_iterator(self):
    #     #with mlxu.open_file(self.config.path, "r") as fin:
    #     blob = bucket.blob(self.config.path)
    #     # The self.config.path should be the data addr after the bucket name,
    #     # such as "cxing/saferdialogues_processed_test.json"
    #     data_str = blob.download_as_string(client=None)
    #     full_data = json.loads(data_str)
    #     data = full_data[self._index]
    #     self._index += 1
    #     if self._index == len(full_data):
    #         self._index = 0
    #     if data is not None:
    #         yield data, self._file_loc, self._index

    def json_iterator(self):
        with mlxu.open_file(self.config.path, "r") as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:  # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index, None
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index, data_source in self.json_iterator():
                yield self.text_processor((example, loc, index, data_source), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(self.json_iterator(), self.config.tokenizer_parallel_batch_size)
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def log_metrics(self, loc, index, average_throughput, accumulated_throughput):
        metrics = {
            "dataset_file_loc": loc,
            "dataset_example_index": index,
            "dataset_total_tokens": self._total_tokens,
            "dataset_accumulated_tps": accumulated_throughput,
            "dataset_average_tps": average_throughput,
        }
        return metrics

    def count_data_source_tokens(self, ds, tokens):
        pass

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index, data_source in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            self.count_data_source_tokens(data_source, tokens)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = self.log_metrics(loc, index, average_throughput, accumulated_throughput)
                batch = {
                    "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch["input_tokens"][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return {
            "config": self.config,
            "index": self._index,
            "file_loc": self._file_loc,
            "total_tokens": self._total_tokens,
        }

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            self.config.update(ConfigDict(state_dict["config"]))
        self._index = state_dict.get("index", self.config.example_index_at_start)
        self._file_loc = state_dict.get("file_loc", self.config.start_seek_loc)
        self._total_tokens = state_dict.get("total_tokens", self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class MultiSourceJsonDataset(JsonDataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ""
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._data_dir = os.path.dirname(self.config.path)
        with mlxu.open_file(self.config.path, "r") as fin:
            data_sources = json.load(fin)
        self._data_sources = [ds["source"] for ds in data_sources]
        self._data_sources_weights = [ds["effective_tokens"] * 10**9 / ds["avg_length"] for ds in data_sources]
        self._indices = {ds: 0 for ds in self._data_sources}
        self._file_locs = {ds: 0 for ds in self._data_sources}
        self._total_tokens_per_source = {ds: 0 for ds in self._data_sources}
        self._total_tokens = 0

        self.json_iterators = self.build_json_iterators()

    def json_iterator(self):
        while True:
            json_iterator = random.choices(self.json_iterators, weights=self._data_sources_weights)[0]
            yield next(json_iterator)

    def build_json_iterators(self):
        def _iterator(ds):
            json_path = os.path.join(self._data_dir, ds)
            with mlxu.open_file(json_path, "r") as fin:
                fin.seek(self._file_locs[ds])
                while True:
                    line = fin.readline()
                    self._file_locs[ds] = fin.tell()
                    if not line:  # Reached EOF
                        self._indices[ds] = 0
                        fin.seek(0)
                        continue

                    data = self.parse_json(line)
                    if data is not None:
                        # JSON parsing succeeded
                        yield data, self._file_locs[ds], self._indices[ds], ds
                    self._indices[ds] += 1

        json_iterators = [_iterator(ds) for ds in self._data_sources]
        return json_iterators

    def count_data_source_tokens(self, ds, tokens):
        self._total_tokens_per_source[ds] += len(tokens)

    def log_metrics(self, loc, index, average_throughput, accumulated_throughput):
        metrics = {
            "dataset_total_tokens": self._total_tokens,
            "dataset_accumulated_tps": accumulated_throughput,
            "dataset_average_tps": average_throughput,
        }
        for ds in self._data_sources:
            ds_name = ds.split(".")[-2]
            metrics[f"dataset_{ds_name}_total_tokens"] = self._total_tokens_per_source[ds]
            metrics[f"dataset_{ds_name}_example_index"] = self._indices[ds]
            metrics[f"dataset_{ds_name}_file_loc"] = self._file_locs[ds]
        return metrics

    def get_state_dict(self):
        return {
            "config": self.config,
            "indices": self._indices,
            "file_locs": self._file_locs,
            "total_tokens": self._total_tokens,
            "total_tokens_per_source": self._total_tokens_per_source,
            "data_sources": self._data_sources,
            "data_sources_weights": self._data_sources_weights,
        }

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            self.config.update(ConfigDict(state_dict["config"]))
        self._indices = state_dict.get("indices")
        self._file_locs = state_dict.get("file_locs")
        self._total_tokens = state_dict.get("total_tokens")
        self._total_tokens_per_source = state_dict.get("total_tokens_per_source")
        self._data_sources = state_dict.get("data_sources")
        self._data_sources_weights = state_dict.get("data_sources_weights")
