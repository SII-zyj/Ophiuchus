#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train ./scripts/SFT/reflective_rejection_fine_tuning/config_reflective.yaml
