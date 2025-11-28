# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


# def build_sam2(
#     config_file,
#     ckpt_path=None,
#     device="cuda",
#     mode="eval",
#     hydra_overrides_extra=[],
#     apply_postprocessing=True,
#     **kwargs,
# ):

#     if apply_postprocessing:
#         hydra_overrides_extra = hydra_overrides_extra.copy()
#         hydra_overrides_extra += [
#             # dynamically fall back to multi-mask if the single mask is not stable
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
#         ]
#     # Read config and init model
#     cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
#     OmegaConf.resolve(cfg)
#     model = instantiate(cfg.model, _recursive_=True)
#     _load_checkpoint(model, ckpt_path)
#     model = model.to(device)
#     if mode == "eval":
#         model.eval()
#     return model
def build_sam2(
    config_file=None,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    cfg=None,  # ✅ 新增：允许外部直接传 cfg
    **kwargs,
):
    """
    构建 SAM2 模型，支持外部或内部 Hydra 配置加载。

    参数：
        config_file: str, 配置文件路径 (.yaml)
        ckpt_path: str, checkpoint 权重路径
        device: str, 默认 "cuda"
        mode: str, "eval" 或 "train"
        hydra_overrides_extra: list, 额外 hydra 参数
        apply_postprocessing: bool, 是否应用 mask decoder 动态后处理
        cfg: 可选，外部已 compose 的 Hydra 配置（推荐）
    """
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    # =============== 后处理设置 ===============
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    # =============== 加载配置 ===============
    if cfg is None:
        # 外部未传 cfg → 内部自行初始化 Hydra
        GlobalHydra.instance().clear()
        config_dir = os.path.dirname(config_file)
        config_name = os.path.basename(config_file).replace(".yaml", "").replace(".yml", "")
        print(f"[build_sam2] ⚙️ 内部初始化 Hydra: {config_dir}/{config_name}")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name, overrides=hydra_overrides_extra)
            print(f"[build_sam2] ✅ 成功加载配置: {config_name}")
    else:
        print(f"[build_sam2] 🔁 使用外部传入的 cfg，不重新 compose")

    # 解析并实例化模型
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    # =============== 加载权重 ===============
    print(f"[build_sam2] ✅ 加载权重: {ckpt_path}")
    _load_checkpoint(model, ckpt_path)

    # =============== 设置设备与模式 ===============
    model = model.to(device)
    if mode == "eval":
        model.eval()

    print(f"[build_sam2] ✅ 模型已构建完成，device={device}, mode={mode}")
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
