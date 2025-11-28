from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Optional
import os
import uuid
import torch
import numpy as np
from PIL import Image
import io
import argparse
import traceback
import uvicorn

from modeling.BaseModel import BaseModel as ModelBaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

parser = argparse.ArgumentParser(description="BiomedParse segmentation service")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--config", type=str, default="configs/biomedparse_inference.yaml")
parser.add_argument("--pretrained", type=str, default="../pretrained/biomedparse_v1.pt")
parser.add_argument("--out_dir", type=str, default="./outputs/biomedparse_masks")
args = parser.parse_args()

app = FastAPI(title="BiomedParse Segmentation Service")


class SegmentRequest(PydanticBaseModel):
    image_path: Optional[str] = Field(None, description="image path")
    image: Optional[str] = Field(None, description="alias of image_path")
    captions: str = Field(..., description="captions")


class BiomedParsePredictor:
    def __init__(self, gpu_id=0, config_path="configs/biomedparse_inference.yaml",
                 pretrained_path="pretrained/biomedparse_v1.pt"):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        self.model = self._build_model()

        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], is_eval=True
            )
        print(f"BiomedParse loaded on {self.device}")

    def _build_model(self):
        try:
            opt = load_opt_from_config_files([self.config_path])
            opt = init_distributed(opt)
            model = ModelBaseModel(opt, build_model(opt)).from_pretrained(self.pretrained_path)
            model = model.eval().to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Model init failed: {str(e)}")

    def segment(self, image: Image.Image, captions: str) -> np.ndarray:
        with torch.no_grad():
            pred_mask = interactive_infer_image(self.model, image, captions)
        return pred_mask.squeeze(0)


biomed_predictor = BiomedParsePredictor(
    gpu_id=args.gpu,
    config_path=args.config,
    pretrained_path=args.pretrained
)

os.makedirs(args.out_dir, exist_ok=True)


def _resolve_image_path(req: SegmentRequest) -> str:
    p = req.image_path or req.image
    if not p:
        raise HTTPException(status_code=400, detail="Missing image_path")
    p = os.path.abspath(p)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail=f"Image not found: {p}")
    if not os.path.isfile(p):
        raise HTTPException(status_code=400, detail=f"Not a file: {p}")
    return p


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    m = mask
    if m.ndim > 2:
        m = np.squeeze(m)
    m = m.astype(np.float32)

    if np.nanmax(m) <= 1.0 and np.nanmin(m) >= 0.0:
        u8 = (m * 255.0).clip(0, 255).astype(np.uint8)
    else:
        u8 = (m > 0.0).astype(np.uint8) * 255
    return u8


@app.post("/biomedparse")
async def segment(req: SegmentRequest):
    try:
        img_path = _resolve_image_path(req)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

        pred_mask = biomed_predictor.segment(image, req.captions)
        mask_u8 = _mask_to_u8(pred_mask)

        out_name = f"biomed_mask_{uuid.uuid4().hex}.png"
        out_path = os.path.abspath(os.path.join(args.out_dir, out_name))
        Image.fromarray(mask_u8, mode="L").save(out_path)

        return JSONResponse(
            content={
                "mask_path": out_path,
                "image_path": img_path,
                "width": image.width,
                "height": image.height,
            }
        )

    except HTTPException as e:
        print(f"HTTP error: {e.detail}")
        traceback.print_exc()
        raise e
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.post("/biomedparse_npy")
async def segment_npy(req: SegmentRequest):
    try:
        img_path = _resolve_image_path(req)
        image = Image.open(img_path).convert("RGB")
        pred_mask = biomed_predictor.segment(image, req.captions)

        buf = io.BytesIO()
        np.save(buf, pred_mask)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=biomed_mask.npy"}
        )
    except HTTPException as e:
        traceback.print_exc()
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "BiomedParse",
        "device": str(biomed_predictor.device),
        "config_path": args.config,
        "pretrained_path": args.pretrained,
        "out_dir": os.path.abspath(args.out_dir),
    }


if __name__ == "__main__":
    print(f"Starting BiomedParse on 0.0.0.0:{args.port} (GPU {args.gpu})")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
