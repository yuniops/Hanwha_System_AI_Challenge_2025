# CondInst + ResNet50 기반 2D Instance Segmentation
# Detectron2 + AdelaiDet + 리소스 사용 모니터링 통합 (모델 학습 전용)

import os
import torch
import psutil
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


#################################################
# 1. 리소스 모니터링 함수
#################################################
def print_resource_usage(step=None):
    if torch.cuda.is_available():
        print(f"[STEP {step}] GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"[STEP {step}] GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"[STEP {step}] CPU Memory Usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")


class ResourceMonitor(HookBase):
    def after_step(self):
        if self.trainer.iter % 10 == 0:
            print_resource_usage(self.trainer.iter)


#################################################
# 2. ZIP 압축 해제 함수
#################################################
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Done.")


#################################################
# 3. COCO 데이터셋 등록
#################################################
def register_datasets():
    register_coco_instances("train2025", {}, "instances_train2025.json", "train2025")
    register_coco_instances("val2025", {}, "instances_val2025.json", "val2025")


#################################################
# 4. Config 설정 함수 (CondInst + ResNet50)
#################################################
def setup_cfg():
    from adet.config import get_cfg as get_adet_cfg
    cfg = get_adet_cfg()
    cfg.merge_from_file("../AdelaiDet/configs/CondInst/Base-CondInst.yaml")
    cfg.DATASETS.TRAIN = ("train2025",)
    cfg.DATASETS.TEST = ("val2025",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.OUTPUT_DIR = "./output_condinst"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


#################################################
# 5. 메인 함수 (학습만 수행)
#################################################
def main():
    extract_zip("train2025.zip", "train2025")
    extract_zip("val2025.zip", "val2025")

    register_datasets()
    cfg = setup_cfg()

    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([ResourceMonitor()])  # 리소스 사용 모니터링 Hook 추가
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("val2025", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "val2025")
    print("[INFO] Running evaluation...")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":
    main()
