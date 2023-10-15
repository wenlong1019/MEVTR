from typing import Union

import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig, AutoTokenizer,
)

import ner_col_data as ner_cd
import pos_col_data as pos_cd
from ner_dataset import NerDataset
from pixel import (
    Modality,
    PangoCairoTextRenderer,
    Split,
    PyGameTextRenderer,
    get_transforms,
    resize_model_embeddings
)
from pos_dataset import PosDataset


class MEVTR_Dataset(Dataset):
    def __init__(self, data_path, visual_encoder, textual_encoder, settings):
        super().__init__()

        self.index_entries = None
        self.settings = settings

        self.visual_values = None
        self.textual_values = None

        if "train" in data_path:
            mode = Split.TRAIN
        elif "dev" in data_path:
            mode = Split.DEV
        elif "test" in data_path:
            mode = Split.TEST

        self.task = settings.task

        print("Loading data from {}".format(data_path))

        self._load_visual_data(mode, visual_encoder)
        self._load_textual_data(mode, textual_encoder)
        self._load_data(data_path)

        print("Done")

    def _load_visual_data(self, mode, visual_encoder):
        # Load text renderer when using image modality and tokenizer when using text modality
        modality = Modality.IMAGE

        renderer_cls = PyGameTextRenderer if self.settings.rendering_backend == "pygame" else PangoCairoTextRenderer

        processor = renderer_cls.from_pretrained(
            self.settings.visual_model_name_or_path,
            fallback_fonts_dir=self.settings.fallback_fonts_dir,
            rgb=self.settings.render_rgb,
        )

        if processor.max_seq_length != self.settings.visual_max_seq_length:
            processor.max_seq_length = self.settings.visual_max_seq_length

        resize_model_embeddings(visual_encoder, processor.max_seq_length)

        # Load dataset
        train_dataset = self.get_dataset(visual_encoder.config,
                                         self.settings.data_dir,
                                         self.settings.visual_max_seq_length,
                                         self.settings.overwrite_cache,
                                         processor, modality, mode)

        self.visual_values = train_dataset.features

    def _load_textual_data(self, mode, textual_encoder):
        # Load text renderer when using image modality and tokenizer when using text modality
        modality = Modality.TEXT

        processor = AutoTokenizer.from_pretrained(
            self.settings.textual_model_name_or_path,
            use_fast=True,
        )

        # Load dataset
        train_dataset = self.get_dataset(textual_encoder.config,
                                         self.settings.data_dir,
                                         self.settings.textual_max_seq_length,
                                         self.settings.overwrite_cache,
                                         processor, modality, mode)

        self.textual_values = train_dataset.features

    def get_dataset(self,
                    config: PretrainedConfig,
                    data_dir, max_seq_length, overwrite_cache,
                    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
                    modality: Modality,
                    split: Split,
                    ):
        kwargs = {}
        if modality == Modality.IMAGE:
            transforms = get_transforms(
                do_resize=True,
                size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
            )
        else:
            transforms = None
            kwargs.update({
                "sep_token_extra": bool(config.model_type in ["roberta"]),
                "cls_token": processor.cls_token,
                "sep_token": processor.sep_token,
                "pad_token": processor.convert_tokens_to_ids([processor.pad_token])[0]
            })

        if self.task == "ner":
            return NerDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                overwrite_cache=overwrite_cache,
                mode=split,
                **kwargs
            )
        elif self.task == "pos":
            return PosDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                overwrite_cache=overwrite_cache,
                mode=split,
                **kwargs
            )

    def _load_data(self, data_path):
        if self.task == "ner":
            data = ner_cd.read_col_data(data_path)
            self.index_entries = []
            for sentence in data:
                self.index_entries.append(ner_cd.IndexEntry(self.settings, sentence))

        elif self.task == "pos":
            data = pos_cd.read_col_data(data_path)
            self.index_entries = []
            for sentence in data:
                self.index_entries.append(pos_cd.IndexEntry(self.settings, sentence))

    def __len__(self):
        return len(self.index_entries)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        entry = self.index_entries[idx]
        visual_data = self.visual_values[idx]
        textual_data = self.textual_values[idx]
        targets = torch.Tensor(entry.targets)
        return (entry._id, targets,
                visual_data["pixel_values"], visual_data["attention_mask"], visual_data["word_starts"],
                textual_data["input_ids"], textual_data["attention_mask"], textual_data["word_starts"],)
