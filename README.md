## Infrared Sensor Instance Segmentation Challenge 2025

기간 : 2025.03.31 ~ 2025.04.30
내용 : 2D instance Segmentation

사이트 : https://www.hscaichallenge.com/

## 가상환경 설정(CONDA)

**step-1: python 3.10 version 가상환경 만들기**
```
conda create -n condinst_env python=3.10 -y
conda activate condinst_env
```

**Step-2: 필수 라이브러리 설치**
```
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```
pycocotools (COCO format 용)
```
pip install cython
pip install pycocotools
```
OpenCV, Pillow, 기타 유틸
```
pip install opencv-python pillow tqdm matplotlib seaborn
pip install scipy scikit-image
```
git clone 받을 때 쓸 경우
```
pip install gitpython
```

**Step-3: 필수 라이브러리 설치**
```
pip install 'git+https://github.com/facebookresearch/detectron2.git@main'
```

**Step-4: AdelaiDet 설치 (CondInst 포함)**
```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```

**Step-5: 설치 확인**
```
python -c "import detectron2; import adet; print('설치 성공')"
```

## 데이터


### train 데이터 - 클래스 별 샘플 수
```
person (1): 13495 instances
car (2): 26932 instances
truck (3): 3001 instances
bus (4): 1308 instances
bicycle (5): 435 instances
bike (6): 1178 instances
extra_vehicle (7): 55 instances
dog (8): 2553 instances
```

### validation 데이터 - 클래스 별 샘플 수
```
person (1): 1388 instances
car (2): 3259 instances
truck (3): 345 instances
bus (4): 198 instances
bicycle (5): 37 instances
bike (6): 181 instances
extra_vehicle (7): 17 instances
dog (8): 63 instances
```



| AP | AP50 | AP75 | APs | APm | APl |
| --- | --- | --- | --- | --- | --- |
| 28.036 | 50.143 | 26.626 | 19.743 | 38.908 | 33.112 |
| [04/21 15:36:08 d2.evaluation.coco_evaluation]: Per-category segm AP: |  |  |  |  |  |
| category | AP | category | AP | category | AP |
| :-------------- | :------- | :----------- | :------- | :----------- | :------- |
| person | 31.075 | car | 54.968 | truck | 46.887 |
| bus | 31.773 | bicycle | 9.348 | bike | 22.296 |
| extra_vehicle | 11.915 | dog | 16.025 |  |  |

