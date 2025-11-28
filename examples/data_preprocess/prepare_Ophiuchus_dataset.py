"""
Preprocess BioMedParse-style RL data for the medical agent environment.

The medical agent expects `env_kwargs` containing the per-episode case metadata
(image paths, question/options, answers, etc.). This script converts the raw
JSON list into a parquet file with those fields populated so `env.reset()` can
load the correct case instead of sampling a default template without images.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_JSON = "./data/example/data_case/biomedparser-RL.json"
DEFAULT_OUT_DIR = "./data/example/data_case/train"


def build_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Construct the single-turn prompt shown to the policy."""
    question = sample.get("question", "")
    options: List[str] = sample.get("options", [])
    question_type = sample.get("question_type", "multiple choice")

    opt_str = "\n".join(options)
    if question_type == "multiple choice":
        content = (
            f"<image>\n{question}\n\nOptions:\n{opt_str}\n\n"
            "Please answer with a single option letter."
        )
    else:
        content = (
            f"<image>\n{question}\n\nOptions:\n{opt_str}\n\n"
            "Please select the correct option."
        )

    return [{"role": "user", "content": content}]


def normalize_image_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")
    return abs_path


def build_env_kwargs(sample: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    """Pack episode kwargs consumed by the medical agent env reset."""
    image_paths = sample.get("image_path") or []
    if not image_paths:
        raise ValueError(f"Sample {idx} ({split}) is missing image_path")

    primary_image = your/path/to(image_paths[0])

    case = {
        "question": sample.get("question", ""),
        "question_type": sample.get("question_type", "multiple choice"),
        "options": sample.get("options", []),
        "answer": sample.get("answer", ""),
        "data_type": sample.get("data_type", "vqa"),
        "source": sample.get("source"),
        "image_path": [primary_image],
        "question_id": sample.get("question_id"),
        "mask_path": sample.get("mask_path"),
        "max_steps": sample.get("max_steps"),
    }

    # Remove None values to keep the payload compact
    return {k: v for k, v in case.items() if v is not None}


def build_row(sample: Dict[str, Any], idx: int, split: str = "train") -> Dict[str, Any]:
    env_kwargs = build_env_kwargs(sample, idx, split)
    prompt = build_prompt(sample)

    images = [{"image": env_kwargs["image_path"][0]}]

    reward_model = {
        "style": "rule",
        "ground_truth": sample.get("answer"),
        "question_type": sample.get("question_type"),
        "options": sample.get("options"),
        "data_type": sample.get("data_type"),
    }

    extra_info = {
        "split": split,
        "index": idx,
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "image_path": sample.get("image_path"),
        "source": sample.get("source"),
    }

    row = {
        "data_source": "medical_agent",
        "prompt": prompt,
        "images": images,
        "ability": "medical",
        "reward_model": reward_model,
        "extra_info": extra_info,
        "env_kwargs": env_kwargs,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Preprocess BioMedParse RL data")
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON, help="Input JSON file")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--split", default="train", help="Split name to tag in extra_info")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [build_row(sample, idx, split=args.split) for idx, sample in enumerate(data)]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)

    out_path = os.path.join(args.out_dir, f"{args.split}.parquet")
    pq.write_table(table, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()