set -x
export HF_CLIP_PATH=/your/path/to/verl-agent/examples/env_server/BiomedParse/BiomedParse-main/openai/clip-vit-base-patch32
export HF_BIOMEDBERT_PATH=/your/path/to/verl-agent/examples/env_server/BiomedParse/BiomedParse-main/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /your/path/to/verl-agent/examples/env_server/BiomedParse/BiomedParse-main
export CUDA_VISIBLE_DEVICES=0
source activate biomedparse

python /your/path/to/verl-agent/examples/env_server/BiomedParse/BiomedParse-main/biomedpaarse-server.py --port 6061
