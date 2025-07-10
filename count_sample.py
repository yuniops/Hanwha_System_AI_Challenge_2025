## instance.json 파일의 클래스별 샘플 수를 구하는 코드임당##
import json
from collections import Counter

# 파일 경로는 실제 파일 경로로 변경해주세요
with open("instances_train2025.json", "r") as f:
    data = json.load(f)

# annotations에서 category_id만 추출
category_ids = [ann["category_id"] for ann in data["annotations"]]

# 개수 세기
counts = Counter(category_ids)

# id -> class name 매핑
id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

# 출력
for cat_id, count in sorted(counts.items()):
    name = id_to_name.get(cat_id, "Unknown")
    print(f"{name} ({cat_id}): {count} instances")