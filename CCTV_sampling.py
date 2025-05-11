import os
import shutil
import json

source_img_dir = r'PRE_YOLOv8_format_data\validation\source\small\legal'
source_label_dir = r'PRE_YOLOv8_format_data\validation\labeling\small\legal'
target_img_dir = r'one-class-dataset\images\val'
target_label_dir = r'one-class-dataset\labels\val'

os.makedirs(target_img_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

max_files = 110
copied_count = 0

for filename in os.listdir(source_label_dir):
    if copied_count >= max_files:
        print(f"{max_files}개 complete")
        break

    if filename.endswith('.json'):
        json_path = os.path.join(source_label_dir, filename)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content_str = json.dumps(data)
                if 'CCTV' in content_str:
                    base = os.path.splitext(filename)[0]
                    image_path = os.path.join(source_img_dir, base + '.jpg')

                    if os.path.exists(image_path):
                        shutil.copy2(json_path, os.path.join(target_label_dir, filename))
                        shutil.copy2(image_path, os.path.join(target_img_dir, base + '.jpg'))
                        copied_count += 1
                        print(f"({copied_count}) 복사 완료: {filename}")
                    else:
                        print(f"이미지 없음: {image_path}")
        except Exception as e:
            print(f"JSON 파싱 실패: {filename} - {e}")
