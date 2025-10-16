import os
import copy
from dataclasses import dataclass
import json
from glob import glob
import random
from typing import Dict, Optional, Sequence, List, Iterator
from operator import itemgetter
from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
import transformers
import tokenizers

import conversation as conversation_lib
from data_utils import DataInfo, annToMask, process_video
from packaging import version

from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices, clips_at_regular_indices, clips_at_random_timestamps, clips_at_regular_timestamps

from open_flamingo.train.any_res_data_utils import process_anyres_image
from open_flamingo.train.data_utils import annToMask, process_video, load_video_frames_with_timestamps

import math
from datetime import datetime
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


IMAGE_FOLDER_DICT_GCP = {
    "allava_laion": "/path/to/datasets/llava-hound-dpo/train_video_and_instruction/allava_laion",
    "allava_vflan": "/path/to/datasets/llava-hound-dpo/train_video_and_instruction/allava_vflan",
    "coco/train2017": "/path/to/datasets/coco/train2017",
    "ocr_vqa": "/path/to/datasets/ocr_vqa",
    "vg": "/path/to/datasets/vg",
    "gqa": "/path/to/datasets/gqa",
    "share_textvqa": "/path/to/datasets/share_textvqa",
    "textvqa": "/path/to/datasets/textvqa",
    'wikiart': "/path/to/datasets/wikiart",
    'sam/images': '/path/to/datasets/sam/images',
    "web-celebrity": "/path/to/datasets/web-celebrity",
    "web-landmark": "/path/to/datasets/web-landmark",
    "llava/llava_pretrain": "/path/to/datasets/llava/llava_pretrain",
    "train2017": "/path/to/datasets/coco/train2017",
}


def get_image_fullpath(image_file):
    image_file_fp = None
    for k, v in IMAGE_FOLDER_DICT_GCP.items():
        if k in image_file:
            image_file_fp = image_file.replace(k, v)
            break
    if image_file_fp is None:
        print(f"File not found: {image_file}")
        exit(0)
    return image_file_fp


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    if image_token_index < 0 or image_token_index >= tokenizer.vocab_size:
        if tokenizer.model_max_length > 2048:
            max_len = 2048
        else:
            max_len = tokenizer.model_max_length
        input_ids = tokenizer(prompt,
                              max_length=max_len - 256,
                              truncation=True).input_ids
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX  # Mask the conv header.
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        print(sentence["value"], flush=True)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        if tokenizer.model_max_length > 2048:
            max_len = 2048
        else:
            max_len = tokenizer.model_max_length
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi_2(
        sources,
        conv_template,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        if tokenizer.model_max_length > 2048:
            max_len = 2048
        else:
            max_len = tokenizer.model_max_length
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len - 256,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.PHI_2

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        rounds_len = len(rounds)
        cur_len = 0  # No <bos> token.
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length - 256:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi_3(
        sources,
        conv_template,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        try:
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        except:
            print("Prompt reading error")
    
    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        # Truncate to 2048 to save memory.
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len - 256,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.PHI_3

    # Mask targets
    sep = conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2 + '\n')
        rounds_len = len(rounds)
        cur_len = 0  # No <bos> token.
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou += conv.sep2 + '\n'
            if sep in rou:
                # assistant round
                round_ids = tokenizer_image_token(rou, tokenizer)
                role_prefix_ids = tokenizer(sep).input_ids
                len_prefix = len(role_prefix_ids)
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif conv.roles[0] in rou:
                # user round
                rou += sep
                if has_image:
                    round_ids = tokenizer_image_token(rou, tokenizer)
                    if i > 0:
                        round_ids = round_ids[1:]  # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len  # All are instructions.
                else:
                    round_ids = tokenizer(rou).input_ids
                    if i > 0:
                        round_ids = round_ids[2:]  # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            else:
                # system round
                round_ids = tokenizer_image_token(rou, tokenizer)
                round_len = len(round_ids)
                instruction_len = round_len  # All are instructions.
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len - 256:  # The input_ids are truncated to this max length.
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
        sources,
        conv_template,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len - 256,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(rounds)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3_instruct(
        sources,
        conv_template,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conv_template.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len - 256,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_INST

    # Mask targets
    sep = f"<|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            rou += conv.sep2
            if sep in rou:
                # assistant round
                round_ids = tokenizer_image_token(rou, tokenizer)
                role_prefix_ids = tokenizer(sep).input_ids
                len_prefix = len(role_prefix_ids)
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif f"<|start_header_id|>{conv.roles[0]}<|end_header_id|>" in rou:
                # user round
                rou += sep
                if has_image:
                    round_ids = tokenizer_image_token(rou, tokenizer)
                    if i > 0:
                        round_ids = round_ids[1:]  # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len  # All are instructions.
                else:
                    round_ids = tokenizer(rou).input_ids
                    if i > 0:
                        round_ids = round_ids[1:]  # Skip the bos tokens
                    round_len = len(round_ids)
                    instruction_len = round_len
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            else:
                # system round
                round_ids = tokenizer_image_token(rou, tokenizer)
                round_len = len(round_ids)
                instruction_len = round_len  # All are instructions.
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len - 256:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(rounds)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len - 256,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < max_len - 256:  # The input_ids are truncated to this max length.
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(rounds)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        conv_template_name: Optional[str] = None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conv_template_name is not None and conv_template_name in conversation_lib.conv_templates.keys():
        # Use the specified preproseccing func.
        conv_template = conversation_lib.conv_templates[conv_template_name]
    else:
        conv_template = conversation_lib.default_conversation

    if conv_template.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conv_template.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conv_template.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conv_template.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conv_template.version.startswith("phi_2"):
        return preprocess_phi_2(sources, conv_template, tokenizer, has_image=has_image)
    if conv_template.version.startswith("phi_3"):
        return preprocess_phi_3(sources, conv_template, tokenizer, has_image=has_image)
    if conv_template.version.startswith("llava_llama_3"):
        return preprocess_llama3(sources, conv_template, tokenizer, has_image=has_image)
    if conv_template.version.startswith("llama_3_instruct"):
        return preprocess_llama3_instruct(sources, conv_template, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)  # Mask human words in the target.

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor,
                 data_args,
                 repeated: bool=False,
                 repeated_size: int=0,
                 ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str) and os.path.isfile(data_path):
            # Load the default 650k data mix.
            list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, str) and os.path.isdir(data_path):
            # Load a custom mixture of data with a list of json files.
            json_lists = glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_lists:
                list_data_dict.extend(json.load(open(json_file, "r")))
        elif isinstance(data_path, Dict):
            list_data_dict = []
            for json_file, n_sample in data_path.items():
                d_json = json.load(open(json_file, "r"))
                random.Random(42).shuffle(d_json)
                list_data_dict.extend(d_json[:n_sample])
        else:
            raise ValueError(f"Unknown data_path type: {data_path}")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template_name = data_args.conv_template_name
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
        self.repeated = repeated
        self.repeated_size = repeated_size

    def __len__(self):
        if self.repeated:
            return self.repeated_size
        else:
            return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample:
                if isinstance(sample['image'], list):
                    cur_len += 384 * len(sample['image'])  # Approximate image token length.
            else:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def _process_single_image(self, image_file) -> Dict[str, torch.Tensor]:
        image_file_fullpath = get_image_fullpath(image_file)
        success = True
        try:
            image = Image.open(image_file_fullpath).convert('RGB')
        except:
            print(f"error opening the file: {image_file_fullpath}")
            success = False
            return success, None, None
        processor = self.image_processor
        img_size = image.size
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.transforms[-1].mean))
            image = processor(image)
        elif self.data_args.image_aspect_ratio == "anyres":
            base_img_size = self.image_processor.transforms[0].size[0]
            image = process_anyres_image(image, processor, [[base_img_size, base_img_size * 2],
                                                            [base_img_size * 2, base_img_size],
                                                            [base_img_size * 2, base_img_size * 2],
                                                            [base_img_size * 3, base_img_size],
                                                            [base_img_size, base_img_size * 3]])
        else:
            image = processor(image)

        return success, image, img_size

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i % len(self.list_data_dict)]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            has_image = True
            image_file = sources[0]['image']
            if isinstance(image_file, list):
                image = []
                img_size = []
                for single_image in image_file:
                    success, image_i, img_size_i = self._process_single_image(single_image)
                    if not success:
                        # Skip the entire sample if one of the images can't be opened.
                        return self.__getitem__(i + 1)
                    image.append(image_i)
                    img_size.append(img_size_i)
            elif isinstance(image_file, str):
                success, image, img_size = self._process_single_image(image_file)
                if not success:
                    # Skip the entire sample if one of the images can't be opened.
                    return self.__getitem__(i + 1)
            else:
                raise NotImplementedError(f"Unknown image_file type: {image_file}")
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            has_image = False
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            conv_template_name=self.conv_template_name)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            if isinstance(image, list):
                data_dict['image'] = image
            elif image.ndim == 4:  # Any-res image patches of a single image.
                data_dict['image'] = image[None, :]
            else:
                data_dict['image'] = image[None, None,
                                     :]  # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = img_size
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.transforms[
                0].size
            data_dict['image'] = torch.zeros(1, 1, 3, crop_size[0], crop_size[
                1])  # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = crop_size
        return data_dict
    
from .video_utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import numpy as np
from . import video_transforms
import cv2

class LazySupervisedDatasetVid(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor,
                 data_args,
                 repeated: bool=False,
                 repeated_size: int=0,
                 ):
        super(LazySupervisedDatasetVid, self).__init__()
        if isinstance(data_path, str) and os.path.isfile(data_path):
            # Load the default 650k data mix.
            list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, str) and os.path.isdir(data_path):
            # Load a custom mixture of data with a list of json files.
            json_lists = glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_lists:
                list_data_dict.extend(json.load(open(json_file, "r")))
        elif isinstance(data_path, Dict):
            list_data_dict = []
            for json_file, n_sample in data_path.items():
                d_json = json.load(open(json_file, "r"))
                random.Random(42).shuffle(d_json)
                list_data_dict.extend(d_json[:n_sample])
        else:
            raise ValueError(f"Unknown data_path type: {data_path}")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template_name = data_args.conv_template_name
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.num_frames = data_args.num_frames
        self.num_timestamp_tokens = data_args.num_timestamp_tokens
        self.min_video_seconds = data_args.min_video_seconds

        self.repeated = repeated
        self.repeated_size = repeated_size
        

    def __len__(self):
        if self.repeated:
            return self.repeated_size
        else:
            return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample:
                if isinstance(sample['image'], list):
                    cur_len += 384 * len(sample['image'])  # Approximate image token length.
            else:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list
    
    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"


    def temporal_random_crop(self, vframes, num_frames, frame_interval):
        temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        total_frames = len(vframes)
        start_frame_ind, end_frame_ind = temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
        video = vframes[frame_indice]
        video_list = []
        for i in range(len(video)):
            video_list.append(torchvision.transforms.functional.to_pil_image(video[i]))
        return video_list
        
    def temporal_random_crop_pil(self, vframes, num_frames, frame_interval):
        temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        total_frames = len(vframes)
        start_frame_ind, end_frame_ind = temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
        video_list = []
        for _indice in frame_indice:
            video_list.append(vframes[_indice])
        return video_list
    
    def _process_single_image(self, image_file) -> Dict[str, torch.Tensor]:
        image_file_fullpath = get_image_fullpath(image_file)
        success = True
        try:
            image = Image.open(image_file_fullpath).convert('RGB')
        except:
            print(f"error opening the file: {image_file_fullpath}")
            success = False
            return success, None, None
        processor = self.image_processor
        img_size = image.size
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.transforms[-1].mean))
            image = processor(image)  # FIXME: whether to take the 0-th item.
        elif self.data_args.image_aspect_ratio == "anyres":
            base_img_size = self.image_processor.transforms[0].size[0]
            image = process_anyres_image(image, processor, [[base_img_size, base_img_size * 2],
                                                            [base_img_size * 2, base_img_size],
                                                            [base_img_size * 2, base_img_size * 2],
                                                            [base_img_size * 3, base_img_size],
                                                            [base_img_size, base_img_size * 3]])
        else:
            image = processor(image)

        return success, image, img_size
    
    def _process_single_video(self, video) -> Dict[str, torch.Tensor]:
        success = True
        processor = self.image_processor
        frames = []
        frames_size = []
        for image in video:
            img_size = image.size
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.transforms[-1].mean))
                image = processor(image)
            elif self.data_args.image_aspect_ratio == "anyres":
                base_img_size = self.image_processor.transforms[0].size[0]
                image = process_anyres_image(image, processor, [[base_img_size, base_img_size * 2],
                                                                [base_img_size * 2, base_img_size],
                                                                [base_img_size * 2, base_img_size * 2],
                                                                [base_img_size * 3, base_img_size],
                                                                [base_img_size, base_img_size * 3]])
            else:
                image = processor(image)
            frames.append(image)
            frames_size.append(img_size)
        if self.data_args.image_aspect_ratio == 'pad':
            frames = torch.stack(frames)
            frames = frames.unsqueeze(1)
        return success, frames, frames_size


    def _process_frames(self, video, image_aspect_ratio='pad') -> Dict[str, torch.Tensor]:
        success = True
        processor = self.image_processor
        frames = []
        frames_size = []
        for image in video:
            img_size = image.size
            if image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.transforms[-1].mean))
                image = processor(image)
            else:
                image = processor(image)
            frames.append(image)
            frames_size.append(img_size)

        frames = torch.stack(frames)
        frames = frames.unsqueeze(1)
        return success, frames, frames_size
    
    def dummy_video(self):
        shape = (self.num_frames, 3, 224, 224)
        vframes = torch.randint(0, 256, shape, dtype=torch.uint8)
        return vframes

    def load_video_torchcodec_v1(self, image_file):

        decoder = VideoDecoder(image_file)
        duration_seconds = decoder.metadata.duration_seconds

        frame_interval = int(math.floor(decoder.metadata.num_frames / self.num_frames))


        clip = clips_at_random_indices(
            decoder,
            num_clips=1,
            num_frames_per_clip=self.num_frames,
            num_indices_between_frames=frame_interval,
        )
            
        return clip.data[0], clip.pts_seconds[0], duration_seconds
    
    
    def time_to_seconds(self, timestamp):
        """Convert timestamp (HH:MM:SS.sss) to seconds."""
        h, m, s = map(float, timestamp.split(":"))
        total_seconds = h * 3600 + m * 60 + s
        return total_seconds

    def load_video_torchcodec_v2(self, image_file, start_time="00:00:00.000", end_time=None, min_duration=0):

        decoder = VideoDecoder(image_file)

        # Convert the start and end times to frame seconds and indices
        start_seconds = self.time_to_seconds(start_time)
        end_seconds = self.time_to_seconds(end_time) if end_time else decoder.metadata.duration_seconds

        duration_seconds = end_seconds - start_seconds
        if duration_seconds < min_duration:
            return -1, None, None
        
        frame_interval = (end_seconds - start_seconds) / self.num_frames


        clip = clips_at_random_timestamps(
            decoder,
            num_clips=1,
            num_frames_per_clip=self.num_frames,
            seconds_between_frames=frame_interval,
            sampling_range_start=start_seconds,
            sampling_range_end=start_seconds+frame_interval,
        )
        return clip.data[0], clip.pts_seconds[0], duration_seconds

    def time_to_frame(self, timestamp, fps):
        """Convert timestamp (HH:MM:SS.sss) to frame index."""
        h, m, s = map(float, timestamp.split(":"))
        total_seconds = h * 3600 + m * 60 + s
        return int(total_seconds * fps)
    
    def load_video_opencv(self, video_path, num_frames=8, start_time="00:00:00.000", end_time=None):
        # Open video file
        cv2_vr = cv2.VideoCapture(video_path)
        if not cv2_vr.isOpened():
            print("Error: Could not open video file.", video_path)
            return -1, None, None
    
        # Get video properties
        fps = cv2_vr.get(cv2.CAP_PROP_FPS)
        total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print("Error: No frames found in video.", video_path)
            return -1, None, None
    
        # Convert start and end times to frame indices
        start_frame = self.time_to_frame(start_time, fps)
        end_frame = self.time_to_frame(end_time, fps) if end_time else total_frames - 1
    
        # Ensure valid frame range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))

        # Compute duration of the extracted segment in seconds
        duration_seconds = (end_frame - start_frame) / fps
    
        # Generate frame indices to sample within the range
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, dtype=int)

        video_data = []
        frame_timestamps = []

        for i, frame_idx in enumerate(frame_id_list):
            # cv2_vr.set(1, frame_idx)
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                # print(f"Warning: Frame at index {frame_idx} could not be read.", image_file)
                if i > 0:
                    video_data.append(video_data[-1])  # Replicate last frame if read fails
                    frame_timestamps.append(frame_timestamps[-1])
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
            video_data.append(frame)

            # Compute timestamp for the frame in seconds
            frame_time = frame_idx / fps
            frame_timestamps.append(frame_time)
    
        cv2_vr.release()
        video_data = [Image.fromarray(f).convert('RGB') for f in video_data]

        if len(video_data) < num_frames:
            print("Error: Certain frames were not read", video_path)
            return -1, None, None
        
        return video_data, frame_timestamps, duration_seconds
    
    def seconds_to_timetoken(self, cur_seconds, duration_seconds, num_timestamp_tokens):
        return round(num_timestamp_tokens * (cur_seconds/duration_seconds))
    
    def timetoken_to_seconds(self, timetoken, duration_seconds, num_timestamp_tokens):
        return round(duration_seconds * (timetoken/num_timestamp_tokens))
   
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i % len(self.list_data_dict)]

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1

        video_cue_duration_in_seconds = 0

        # --  with mask
        if ('path' in sources[0]) and ('annotation' in sources[0]):  # masks
            has_image = True

            # mask_single_frame = False
            if 'videorefer' in sources[0]:
                mask_single_frame = False
            else:
                mask_single_frame = True
            
            if not mask_single_frame:
                ann_indices = []
                frame_nums = 1
                all_frames = set()
                
                for ann in sources[0]['annotation']:
                    all_frames.update(list(ann.keys()))
                all_frames = list(all_frames)
                frame_nums = len(all_frames)
    
                # sort frame ids
                all_frames = [int(f) for f in all_frames]
                all_frames = sorted(all_frames)
                all_frames = [str(f) for f in all_frames]
                
                for ann in sources[0]['annotation']:
                    frame_list = list(ann.keys())
    
                    # sort frame ids
                    frame_list = [int(f) for f in frame_list]
                    frame_list = sorted(frame_list)
                    frame_list = [str(f) for f in frame_list]
    
                    indices = []
                    for frame in frame_list:
                        indices.append(all_frames.index(frame))
                    ann_indices.append(indices)
                 
                all_frames = [int(f) for f in all_frames]
            else:  # select a random single frame for mask referring
                ann_indices = []
                frame_nums = 1
                all_frames = set()

                selected_frame_per_obj = []
                
                for ann in sources[0]['annotation']: # for each object mask annotation
                    if len(all_frames) == 0:
                        selected_frame = random.sample(list(ann.keys()),1)[0]
                    else:
                        overlapped_frames = all_frames & set(ann.keys())
                        if not overlapped_frames:
                            selected_frame = random.sample(list(ann.keys()),1)[0]
                        else:
                            selected_frame = random.sample(list(overlapped_frames),1)[0]
                    all_frames.add(selected_frame)
                    selected_frame_per_obj.append(selected_frame)
                            
                all_frames = list(all_frames)
                frame_nums = len(all_frames)
    
                # sort frame ids
                all_frames = [int(f) for f in all_frames]
                all_frames = sorted(all_frames)
                all_frames = [str(f) for f in all_frames]

                obj_idx = 0
                for ann in sources[0]['annotation']:
                    indices = [all_frames.index(selected_frame_per_obj[obj_idx])]
                    ann_indices.append(indices)

                    obj_idx += 1
                 
                all_frames = [int(f) for f in all_frames]
                
            try:
                if 'videorefer' in sources[0]:
                    video, mask_frames, height, width = process_video(
                        sources[0]['path'], num_frames=self.num_frames, frame_idx=all_frames, MAX_FRAMES=self.num_frames) 
                else:
                    video, _, _, video_cue_duration_in_seconds = load_video_frames_with_timestamps(
                        sources[0]['path'], n_frames=self.num_frames, start_time="00:00:00.000", end_time=None)
                    height, width = np.array(video[0]).shape[:2]
                    
                    if self.num_frames == frame_nums:
                        mask_frames = video
                    else:
                        mask_frames = [video[frame_idx] for frame_idx in all_frames]
            except Exception as e:
                video = -1
                print(f"An error occurred: {e}")
            
            if video == -1:
                success = False
                frames = None
                frames_size = None
            else:
                
                success, frames, frames_size = self._process_single_video(video)
            
            if not success:
                print("Error reading the video (mask refer) (videorefer: {}) and going to skip it: {}".format(
                    'videorefer' in sources[0], sources[0]['path']))
                
                # Skip the entire sample
                return self.__getitem__(i + 1)
                
            
            masks = []
            if 'height' in sources[0]:
                h = sources[0]['height']
                w = sources[0]['width']
            else:
                h = None
                w = None

            if not mask_single_frame:
                for ann in sources[0]['annotation']:
                    frame_list = list(ann.keys())
    
                    # sort frame ids
                    frame_list = [int(f) for f in frame_list]
                    frame_list = sorted(frame_list)
                    frame_list = [str(f) for f in frame_list]
                    
                    for ann_idx in frame_list:
                        
                        try:
                            mask = annToMask(ann[ann_idx]['segmentation'], h, w)
                        except:
                            mask = np.zeros((height, width))
                        masks.append(mask)
            else:
                obj_idx = 0
                for ann in sources[0]['annotation']:
                    try:
                        mask = annToMask(
                            ann[selected_frame_per_obj[obj_idx]]['segmentation'], h, w)
                    except:
                        mask = np.zeros((height, width))
                    masks.append(mask)
    
                    obj_idx += 1
                
            mask_frame_count = 0
            for obj_idx in range(len(ann_indices)):
                mask_frame_count += len(ann_indices[obj_idx])
            assert len(masks) == mask_frame_count
            
            masks = np.array(masks)      

        # --  no mask
        elif ('path' in sources[0]) and ('annotation' not in sources[0]):
            try:
                has_image = True
                image_file = sources[0]['path']
                if not os.path.isdir(image_file): # .mp4 or avi and others 
                    if 'clip_timestamps' in sources[0]:
                        video, frame_time_in_seconds, video_cue_duration_in_seconds = self.load_video_torchcodec_v2(
                            image_file,
                            start_time=sources[0]['clip_timestamps'][0],
                            end_time=sources[0]['clip_timestamps'][1])
                    else:
                        video, frame_time_in_seconds, video_cue_duration_in_seconds = self.load_video_torchcodec_v1(image_file)
                    
                    # Sampling video frames
                    total_frames = len(video)
                    frame_interval = int(math.floor(total_frames / self.num_frames))
                    video = self.temporal_random_crop(video, self.num_frames, frame_interval)
                    
                else:
                    fpath_list = glob(os.path.join(image_file, '*.jpeg'))
                    vframes = [Image.open(image_path).convert('RGB') for image_path in fpath_list]
                    total_frames = len(vframes)
                    frame_interval = int(math.floor(total_frames / self.num_frames))
                    video = self.temporal_random_crop_pil(vframes, self.num_frames, frame_interval)
    
                if video == -1:
                    success = False
                    frames = None
                    frames_size = None
                else:
                    success, frames, frames_size = self._process_single_video(video)
                
                
            except:
                success = False
                frames = None
                frames_size = None
        
            if not success:
                try:
                    has_image = True
                    image_file = sources[0]['path']
                    if not os.path.isdir(image_file): # .mp4 or avi and others 
                        if 'clip_timestamps' in sources[0]:
                            video, frame_time_in_seconds, video_cue_duration_in_seconds = self.load_video_opencv(image_file, 
                                                           num_frames=self.num_frames,                                                    
                                                           start_time=sources[0]['clip_timestamps'][0], 
                                                           end_time=sources[0]['clip_timestamps'][1])
                        else:
                            video, frame_time_in_seconds, video_cue_duration_in_seconds = self.load_video_opencv(
                                image_file, num_frames=self.num_frames)
                        
                    else:
                        fpath_list = glob(os.path.join(image_file, '*.jpeg'))
                        vframes = [Image.open(image_path).convert('RGB') for image_path in fpath_list]
                        total_frames = len(vframes)
                        frame_interval = int(math.floor(total_frames / self.num_frames))
                        video = self.temporal_random_crop_pil(vframes, self.num_frames, frame_interval)
        
                    if video == -1:
                        success = False
                        frames = None
                        frames_size = None
                    else:
                        success, frames, frames_size = self._process_single_video(video)
                        
                        
                except:
                    success = False
                    frames = None
                    frames_size = None
                    
            if not success:
                print("Error reading the video (eventually) and going to skip it: {}".format(image_file))
                # Skip the entire sample
                return self.__getitem__(i + 1)
        
        else:
            has_image = False
            sources = copy.deepcopy([e["conversations"] for e in sources])
            raise NotImplementedError(f"No Videos")
        
        
        if ('start_time' in sources[0]) or ('end_time' in sources[0]):
            
            if not video_cue_duration_in_seconds:
                _, _, video_cue_duration_in_seconds = self.load_video_opencv(sources[0]['path'])
                
            start_second = self.time_to_seconds(sources[0]['start_time'])
            end_second = self.time_to_seconds(sources[0]['end_time'])
            
            start_time = self.seconds_to_timetoken(start_second, video_cue_duration_in_seconds, self.num_timestamp_tokens-1)
            end_time = self.seconds_to_timetoken(end_second, video_cue_duration_in_seconds, self.num_timestamp_tokens-1)

            # if (end_time - start_time) < 1:
            #     print("Time referring is too short and going to skip it: {}".format(image_file))
            #     return self.__getitem__(i + 1)
                
            sources[0]['conversations'][0]['value'] = sources[0]['conversations'][0]['value'].replace(
                '<start_time>', '<{}>'.format(start_time))
            sources[0]['conversations'][0]['value'] = sources[0]['conversations'][0]['value'].replace(
                '<end_time>', '<{}>'.format(end_time))
            
            sources[0]['conversations'][1]['value'] = sources[0]['conversations'][1]['value'].replace(
                '<start_time>', '<{}>'.format(start_time))
            sources[0]['conversations'][1]['value'] = sources[0]['conversations'][1]['value'].replace(
                '<end_time>', '<{}>'.format(end_time))
        
        
        data_dict = preprocess(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.tokenizer,
            has_image=has_image,
            conv_template_name=self.conv_template_name)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['sample_id'] = '{}'.format(i)


        if ('path' in sources[0]) and ('annotation' in sources[0]):  # masks
            mask_frames_success, mask_frames, mask_frames_size = self._process_frames(
                mask_frames, image_aspect_ratio='square')
            
            data_dict['mask_frames'] = mask_frames
            data_dict['frame_nums']= frame_nums
            data_dict['masks'] = torch.Tensor(masks)
            data_dict['ann_indices'] = ann_indices
        else:
            data_dict['mask_frames'] = None
            data_dict['frame_nums']= None
            data_dict['masks'] = None
            data_dict['ann_indices'] = None            
        
        # image exist in the data
        if has_image:
            if isinstance(frames, list):
                data_dict['image'] = [_image[:] for _image in frames] # image
            elif frames.ndim == 5:
                data_dict['image'] = frames
            else:
                raise NotImplementedError(f"No Videos")
                
            data_dict['image_size'] = frames_size
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.transforms[0].size
            data_dict['image'] = torch.zeros(1, 1, 3, crop_size[0], crop_size[1])  
            # Expand dims with [T_img, F] to be compatible with flamingo-like vision encoding.
            data_dict['image_size'] = crop_size
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_size = [instance['image_size'] for instance in instances]
            sample_id = [instance['sample_id'] for instance in instances]
            batch['image_size'] = image_size
            batch['sample_id'] = sample_id

            batch['mask_frames'] = [instance['mask_frames'] for instance in instances]
            batch['frame_nums'] = [instance['frame_nums'] for instance in instances]
            batch['masks'] = [instance['masks'] for instance in instances]
            batch['ann_indices'] = [instance['ann_indices'] for instance in instances]

            if any(isinstance(x, list) for x in images):
                images_list = []
                for x in images:
                    if isinstance(x, torch.Tensor):
                        images_list.append([x])
                    elif isinstance(x, list):
                        images_list.append(x)
                    else:
                        raise NotImplementedError(f"Unknown data type: {x}")
                image_size_list = []
                for x in image_size:
                    if not isinstance(x, list):
                        image_size_list.append([x])
                    else:
                        image_size_list.append(x)
                batch['images'] = images_list
                batch['image_size'] = image_size_list
            elif all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset.py

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                image_processor,
                                data_args,
                                repeated: bool=False,
                                repeated_size: int=0) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if "image" in data_args.data_path:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            image_processor=image_processor,
                                            data_args=data_args,
                                            repeated=repeated,
                                            repeated_size=repeated_size)
    else:
        train_dataset = LazySupervisedDatasetVid(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            image_processor=image_processor,
                                            data_args=data_args,
                                            repeated=repeated,
                                            repeated_size=repeated_size)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if data_args.data_sampler_group_by_length:
        # Use length grouped sampler for more balanced GPU usages.
        lengths = train_dataset.modality_lengths
        sampler_inner = LengthGroupedSampler(
            data_args.batch_size,
            world_size=data_args.world_size * data_args.gradient_accumulation_steps,
            lengths=lengths,
            group_by_modality=True,
        )
        sampler = DistributedSamplerWrapper(
            sampler=sampler_inner, num_replicas=data_args.world_size, rank=data_args.rank, shuffle=True
        )
    else:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )
    
    data_loader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=data_collator,
    )
    return DataInfo(
        name='llava-mix',
        dataloader=data_loader,
        batch_size=data_args.batch_size,
        loss_multiplier=1.0,
        shared_epoch=None,
        sampler=sampler,
    ), len(train_dataset)

