import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.events import get_event_storage
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import zipfile
import matplotlib.pyplot as plt


# 1. 압축 해제
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Done.")

# 2. COCO 데이터셋 등록
def register_datasets():
    register_coco_instances("train2025", {}, "instances_train2025.json", "train2025")
    register_coco_instances("val2025", {}, "instances_val2025.json", "val2025")

# 3. 설정 함수
def setup_cfg():
    from adet.config import get_cfg as get_adet_cfg
    cfg = get_adet_cfg()
    
    cfg.MODEL.FCOS.set_new_allowed(True)  # ← 새 key 허용 ## (여기 추가)
    
    cfg.merge_from_file("../AdelaiDet/configs/CondInst/MS_R_101_BiFPN_3x_sem.yaml")

    # 수동 등록 (cfg에 필드 강제 생성)
    cfg.MODEL.FCOS.NUM_CLASSES = 8 ## (여기 추가)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    
    cfg.MODEL.FCOS.SAMPLES_PER_CLASS = [13495, 26932, 3001, 1308, 435, 1178, 55, 2553]
    cfg.MODEL.FCOS.CB_LOSS_BETA = 0.9999
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.5
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0
    
    # Dataset
    cfg.DATASETS.TRAIN = ("train2025",)
    cfg.DATASETS.TEST = ("val2025",)
    # cfg.DATASETS.TRAIN = ("train2025", "val2025") ## validation도 학습시키기! 
    cfg.DATASETS.TEST = ()  # 평가용 없음
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.MASK_FORMAT = "bitmask"

    # 모델 설정
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    

    # Optimizer
    cfg.SOLVER.IMS_PER_BATCH = 20
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.STEPS = []  # step decay 사용 안 함
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = 100

    # Validation 평가 주기
    cfg.TEST.EVAL_PERIOD = 500 ## 1000인데, 비활성화

    # 출력 디렉토리
    cfg.OUTPUT_DIR = "./output_condinst_r101_bifpn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

# 4. Trainer with best model saving
class BestModelTrainer(DefaultTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.eval_ap = []       # segm/AP 저장 리스트
        self.eval_iters = []    # 해당 iteration 저장 리스트
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        # ✅ best model 저장
        hooks_list.insert(-1, hooks.BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "segm/AP",
            mode="max",
        ))

        # ✅ segm/AP 추적 및 그래프 저장 hook
        def _after_eval_callback():
            eval_result = flatten_results_dict(self.storage.latest())
            segm_ap = eval_result.get("segm/AP", None)
            if segm_ap is not None:
                self.eval_iters.append(self.iter)
                self.eval_ap.append(segm_ap)
                self._save_eval_plot()

        hooks_list.append(hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, _after_eval_callback))

        return hooks_list

    # ✅ 500 iteration마다 total_loss 로그 출력
    def after_step(self):
        super().after_step()
        storage = get_event_storage()
        if self.iter % 500 == 0:
            loss = storage.history("total_loss").latest()
            print(f"[INFO] Iteration {self.iter}: Total Loss = {loss:.4f}")

    # ✅ segm/AP 그래프 저장
    def _save_eval_plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.eval_iters, self.eval_ap, marker='o', linestyle='-')
        plt.xlabel("Iteration")
        plt.ylabel("segm/AP")
        plt.title("segm/AP over Training")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(self.cfg.OUTPUT_DIR, "segm_ap_curve.png")
        plt.savefig(out_path)
        plt.close()

# 5. 메인 실행
def main():
    extract_zip("train2025.zip", "train2025")
    extract_zip("val2025.zip", "val2025")

    register_datasets()
    cfg = setup_cfg()

    trainer = BestModelTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("val2025", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "val2025")
    print("[INFO] Running evaluation...")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":
    main()