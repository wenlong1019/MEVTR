import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
from PIL import Image
from filelock import FileLock
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, is_torch_available

from pixel.data.rendering import PyGameTextRenderer, PangoCairoTextRenderer
from pixel.utils import Modality, Split, get_attention_mask


@dataclass
class PosInputExample:
    words: List[str]


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset


    class PosDataset(Dataset):

        features: List[Dict[str, Union[int, torch.Tensor]]]

        def __init__(
                self,
                data_dir: str,
                processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
                modality: Modality,
                transforms: Optional[Callable] = None,
                max_seq_length: Optional[int] = None,
                overwrite_cache=False,
                mode: Split = Split.TRAIN,
                **kwargs
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, processor.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    self.examples = read_examples_from_file(data_dir, mode)
                    self.features = torch.load(cached_features_file)
                else:
                    self.examples = read_examples_from_file(data_dir, mode)
                    examples_to_features_fn = _get_examples_to_features_fn(modality)
                    self.features = examples_to_features_fn(
                        examples=self.examples,
                        max_seq_length=max_seq_length,
                        processor=processor,
                        transforms=transforms,
                        **kwargs
                    )
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> Dict[str, Union[int, torch.Tensor]]:
            return self.features[i]


def get_file(data_dir: str, mode: Union[Split, str]) -> Optional[str]:
    if isinstance(mode, Split):
        mode = mode.value
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    fp = os.path.join(data_dir, f"*{mode}.conllu")
    _fp = glob.glob(fp)
    if len(_fp) == 1:
        return _fp[0]
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[PosInputExample]:
    file_path = get_file(data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(PosInputExample(words=words))
                    words = []
            if tok[0].isdigit():
                word = tok[1]
                words.append(word)
        if words:
            examples.append(PosInputExample(words=words))
    return examples


def _get_examples_to_features_fn(modality: Modality):
    if modality == Modality.IMAGE:
        return convert_examples_to_image_features
    if modality == Modality.TEXT:
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")


def convert_examples_to_image_features(
        examples: List[PosInputExample],
        max_seq_length: int,
        processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
        transforms: Optional[Callable] = None,
        **kwargs
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `Dict` containing image features"""

    features = []
    for (ex_index, example) in enumerate(examples):
        encoding = processor(example.words)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        # sanity check lengths
        assert len(attention_mask) == max_seq_length

        features.append({"pixel_values": pixel_values,
                         "attention_mask": attention_mask,
                         "word_starts": word_starts})

    return features


def convert_examples_to_text_features(
        examples: List[PosInputExample],
        max_seq_length: int,
        processor: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        **kwargs,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        word_starts = [0]
        word_cnt = 1
        for word in example.words:
            word_tokens = processor.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ["▁"]
            word_starts.append(word_cnt)
            word_cnt = word_cnt + len(word_tokens)
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = processor.num_special_tokens_to_add() + (1 if sep_token_extra else 0)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = processor.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length

        assert len(word_starts) == len(example.words) + 1
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_starts": word_starts,
            }
        )
    return features
