from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .transformer import *
from .build import *


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)

import os
from transformers import CLIPTokenizer, CLIPTokenizerFast, AutoTokenizer

def _resolve_hf_path_or_id(default_id: str, config_value: str = None) -> str:
    """
    允许三种来源（优先级从高到低）：
    1) 环境变量 HF_CLIP_PATH（你可以指向本地目录）
    2) config_encoder['PRETRAINED_TOKENIZER']
    3) 默认的 HuggingFace repo id
    """
    env_path = os.environ.get("HF_CLIP_PATH", "").strip()
    if env_path:
        return env_path
    if config_value:
        return config_value
    return default_id

def _local_only_flag() -> bool:
    """
    如果设置了 TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE 或 HF_LOCAL_FILES_ONLY，则强制只读本地文件。
    """
    for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_LOCAL_FILES_ONLY"):
        v = os.environ.get(k, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    local_only = _local_only_flag()

    if config_encoder["TOKENIZER"] == "clip":
        cfg = config_encoder.get("PRETRAINED_TOKENIZER", None)
        pretrained_tokenizer = _resolve_hf_path_or_id("openai/clip-vit-base-patch32", cfg)
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_tokenizer,
            local_files_only=local_only,
        )
        tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})

    elif config_encoder["TOKENIZER"] == "clip-fast":
        cfg = config_encoder.get("PRETRAINED_TOKENIZER", None)
        pretrained_tokenizer = _resolve_hf_path_or_id("openai/clip-vit-base-patch32", cfg)
        tokenizer = CLIPTokenizerFast.from_pretrained(
            pretrained_tokenizer,
            from_slow=True,
            local_files_only=local_only,
        )

    elif config_encoder["TOKENIZER"] == "biomed-clip":
        cfg = config_encoder.get("PRETRAINED_TOKENIZER", None)
        pretrained = os.environ.get("HF_BIOMEDBERT_PATH", "").strip() or cfg \
            or "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=local_only)

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config_encoder["TOKENIZER"],
            local_files_only=local_only,
        )

    return tokenizer
