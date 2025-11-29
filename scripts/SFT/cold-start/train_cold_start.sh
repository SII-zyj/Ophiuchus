#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train ./scripts/SFT/cold-start/my_qwen2_5vl_lora_sft.yaml
