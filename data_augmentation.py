import os
import json
import cv2
import albumentations as A

# 경로 설정
json_folder = "/content/Big_data_overloaded/big_label"
image_base_dir = "/content/data_overloaded/과적차량데이터/1.Training/원천데이터/TS1.대형차/불법차량"
target_img_dir = "rere_augmented/images"
target_label_dir = "rere_augmented/labels"

os.makedirs(target_img_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

repeat = 2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomCrop(width=256, height=256, p=0.3),
    A.PadIfNeeded(min_height=512, min_width=512, p=0.3),
    A.GaussNoise(p=0.2)
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

label_map = {"불법차량": 0}
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)

    # JSON 파일 오류 시 건너뛰기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[JSON 오류] {json_file}: {e}")
        continue

    for file_data in json_data.get("FILE", []):
        file_name = file_data.get("FILE_NAME")
        if not file_name:
            continue

        img_path = os.path.join(image_base_dir, file_name)

        # label 데이터 이미지 데이터 대조 후 없으면 건너뛰기
        if not os.path.exists(img_path):
            print(f"[이미지 없음] {file_name}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[이미지 읽기 실패 건너뜀] {file_name}")
            continue

        h, w = img.shape[:2]
        bboxes = []
        class_labels = []

        for item in file_data.get("ITEMS", []):
            box = list(map(float, item.get("BOX", "0,0,0,0").split(",")))
            class_name = item.get("PACKAGE", "")
            if class_name not in label_map:
                continue
            bboxes.append(box)
            class_labels.append(label_map[class_name])

        for i in range(repeat):
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            # Augmented 이미지 저장
            base_name = os.path.splitext(file_name)[0]
            aug_img_name = f"{base_name}_aug{i}.jpg"
            cv2.imwrite(os.path.join(target_img_dir, aug_img_name), aug_img)

            # JSON 파일 작성
            new_json_data = {
                "FILE": [
                    {
                        **file_data,
                        "FILE_NAME": aug_img_name,
                        "ITEMS": []
                    }
                ]
            }

            for box, cls in zip(aug_boxes, aug_classes):
                x, y, bw, bh = box
                box_str = f"{int(x)},{int(y)},{int(bw)},{int(bh)}"
                class_name = [k for k, v in label_map.items() if v == cls][0]

                new_item = {     
                    "DRAWING": "Box",
                    "SEGMENT": "대형차",
                    "BOX": box_str,
                    "POLYGON": "",
                    "PACKAGE": class_name,
                    "CLASS": "적재불량",
                    "COVER": "덮개개방",
                    "COURSE": "후면좌측",
                    "CURVE": "정상주행"
                }

                new_json_data["FILE"][0]["ITEMS"].append(new_item)

            aug_json_name = f"{base_name}_aug{i}.json"
            with open(os.path.join(target_label_dir, aug_json_name), 'w', encoding='utf-8') as f:
                json.dump(new_json_data, f, indent=2, ensure_ascii=False)