set -x
cd /your/path/to/verl-agent/examples/env_server/sam2
export CUDA_VISIBLE_DEVICES=0
source activate sam2

python -m sam2_server.sam2_server
