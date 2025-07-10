# test_condinst.py

import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from fvcore.nn import FlopCountAnalysis
from pycocotools import mask as mask_utils


def setup_cfg_for_inference():
    from adet.config import get_cfg as get_adet_cfg
    cfg = get_adet_cfg()
    # cfg.merge_from_file("../AdelaiDet/configs/CondInst/Base-CondInst.yaml")
    
    # 🔧 여기에 추가
    cfg.MODEL.FCOS.set_new_allowed(True)
    cfg.MODEL.FCOS.NUM_CLASSES = 8 ## (여기 추가)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.FCOS.SAMPLES_PER_CLASS = [13495, 26932, 3001, 1308, 435, 1178, 55, 2553]
    cfg.MODEL.FCOS.CB_LOSS_BETA = 0.9999
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.5
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0
    
    
    cfg.merge_from_file("../AdelaiDet/configs/CondInst/MS_R_50_3x_sem.yaml")
    cfg.MODEL.WEIGHTS = "./output_condinst_semantic/model_final_val.pth"
    # cfg.MODEL.WEIGHTS = "./output_condinst_r101/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # 디폴트는 0.5임.
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


# def predict_on_test(predictor, test_root, output_json):
#     results = []
#     test_files = sorted([f for f in os.listdir(test_root) if f.endswith(".png")])
#     for file_name in tqdm(test_files):
#         image_path = os.path.join(test_root, file_name)
#         image = cv2.imread(image_path)
#         height, width = image.shape[:2]

#         # Detectron2의 입력 형식으로 변환
#         image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float().cuda()
#         input_tensor = {
#             "image": image_tensor,
#             "height": height,
#             "width": width
#         }

#         with torch.no_grad():
#             # FLOPs는 이미지 1개 기준 고정 값 사용
#             flops = 82.92  # ← 측정한 GFLOPs 값 직접 입력(resnet50 - 60.13 / resnet101 - 82.92)
#             torch.cuda.synchronize()
#             start_event = torch.cuda.Event(enable_timing=True)
#             end_event = torch.cuda.Event(enable_timing=True)
#             start_event.record()
#             outputs = predictor(image)
#             end_event.record()
#             torch.cuda.synchronize()
#             inference_time = int(start_event.elapsed_time(end_event))  # ms

#         instances = outputs["instances"].to("cpu")
#         predictions = []
#         for i in range(len(instances)):
#             score = float(instances.scores[i])
#             if score < 0.5:
#                 continue
#             bbox = instances.pred_boxes[i].tensor.numpy().tolist()[0]
#             mask = instances.pred_masks[i].numpy().astype(np.uint8)
#             area = int(mask.sum())
#             rle = mask_utils.encode(np.asfortranarray(mask))
#             rle["counts"] = rle["counts"].decode("utf-8")

#             pred = {
#                 "score": score,
#                 "category_id": int(instances.pred_classes[i]) + 1,
#                 "bbox": bbox,
#                 "segmentation": {
#                     "size": [height, width],
#                     "counts": rle["counts"]
#                 },
#                 "area": area,
#                 "iscrowd": 0,
#                 "id": i + 1
#             }
#             predictions.append(pred)

#         results.append({
#             "time": inference_time,
#             "flops": flops,
#             "input_shape": [1, 3, height, width],
#             "file_name": file_name,
#             "prediction": predictions
#         })

#     output_dict = {
#         "info": {
#             "device_name": "NVIDIA GeForce RTX 4090",
#             "device_id": 14
#         },
#         "results": results
#     }

#     with open(output_json, 'w') as f:
#         json.dump(output_dict, f)
#     print(f"[INFO] Test results saved to {output_json}")
def predict_on_test(predictor, test_root, output_json):
    results = []
    test_files = sorted([f for f in os.listdir(test_root) if f.endswith(".png")])
    flops = int(132.19)  # GFLOPs 단위 정수형으로 설정 (59.07 Gflops - fvcore 기준) 
    unique_id = 1  # 예측 객체 고유 ID 시작값

    for file_name in tqdm(test_files):
        image_path = os.path.join(test_root, file_name)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        with torch.no_grad():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = predictor(image)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = int(start_event.elapsed_time(end_event))  # ms

        instances = outputs["instances"].to("cpu")
        predictions = []

        for i in range(len(instances)):
            score = float(instances.scores[i])
            if score < predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                continue

            bbox_tensor = instances.pred_boxes[i].tensor.numpy().tolist()[0]
            x, y, w, h = bbox_tensor[0], bbox_tensor[1], bbox_tensor[2] - bbox_tensor[0], bbox_tensor[3] - bbox_tensor[1]
            bbox = [x, y, x + w, y + h]  # bottom-right 형식

            mask = instances.pred_masks[i].numpy().astype(np.uint8)
            area = int(mask.sum())
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            pred = {
                "score": score,
                "category_id": int(instances.pred_classes[i]) + 1,
                "segmentation": {
                    "size": [height, width],
                    "counts": rle["counts"]
                },
                "area": area,
                "bbox": bbox,
                "id": unique_id,
                "iscrowd": 0
            }
            unique_id += 1
            predictions.append(pred)

        results.append({
            "time": inference_time,
            "flops": flops,
            "input_shape": [1, 3, height, width],
            "file_name": file_name,
            "prediction": predictions
        })

    output_dict = {
        "info": {
            "device_name": "NVIDIA GeForce RTX 4090",  # 실제 환경에 맞게 수정 가능
            "device_id": 14
        },
        "results": results
    }

    with open(output_json, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print(f"[INFO] Test results saved to {output_json}")

def main():
    test_root = "test2025"
    register_coco_instances("test2025", {}, "dummy.json", test_root)  # dummy registration
    cfg = setup_cfg_for_inference()
    predictor = DefaultPredictor(cfg)
    predict_on_test(predictor, test_root, "2025_prediction_test.txt")  # 제출 형식에 맞춰 .txt 확장자


if __name__ == "__main__":
    main()