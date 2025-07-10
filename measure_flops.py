import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from adet.config import get_cfg as get_adet_cfg
from torch.profiler import profile, ProfilerActivity

# 1. 모델 설정 및 초기화
def setup_model():
    cfg = get_adet_cfg()
    cfg.MODEL.FCOS.set_new_allowed(True)
    cfg.MODEL.FCOS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.FCOS.SAMPLES_PER_CLASS = [13495, 26932, 3001, 1308, 435, 1178, 55, 2553]
    cfg.MODEL.FCOS.CB_LOSS_BETA = 0.9999
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.5
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0
    cfg.merge_from_file("../AdelaiDet/configs/CondInst/MS_R_50_3x_sem.yaml")
    cfg.MODEL.WEIGHTS = "./output_condinst_semantic/model_final_val.pth"
    cfg.MODEL.DEVICE = "cuda"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().cuda()
    return model

# 2. Detectron2 → torch.profiler 형식에 맞는 래퍼
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model([{"image": x}])[0]["instances"].pred_boxes.tensor

# 3. torch.profiler로 FLOPs 측정
def measure_flops_with_profiler(model, image_dir, max_images=10):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[:max_images]
    wrapped_model = WrappedModel(model).cuda()
    wrapped_model.eval()

    total_flops = 0
    valid_count = 0

    for img_name in tqdm(image_files, desc="Measuring FLOPs (torch.profiler)"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img_tensor = torch.as_tensor(img.transpose(2, 0, 1)).float().unsqueeze(0).cuda()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True
        ) as prof:
            wrapped_model(img_tensor[0])

        # FLOPs 합산
        flops = 0
        for evt in prof.key_averages():
            if hasattr(evt, "flops") and evt.flops is not None:
                flops += evt.flops

        if flops > 0:
            gflops = flops / 1e9
            print(f"\nFLOPs for {img_name}: {gflops:.2f} GFLOPs")
            total_flops += gflops
            valid_count += 1
        else:
            print(f"\nFLOPs not available for {img_name}")

    if valid_count > 0:
        avg_flops = total_flops / valid_count
        print(f"\nAverage FLOPs over {valid_count} images: {avg_flops:.2f} GFLOPs")
    else:
        print("No valid FLOPs data collected.")

# 4. 실행
if __name__ == "__main__":
    model = setup_model()
    measure_flops_with_profiler(model, "test2025", max_images=10)
