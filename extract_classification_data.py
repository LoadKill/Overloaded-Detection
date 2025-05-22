import os
import cv2


# 입력 폴더
image_dir = r'YOLOv8_twoClass_format_data\images\val'
label_dir = r'YOLOv8_twoClass_format_data\labels\val'

# 출력 폴더
output_base = 'classification_data/test'
legal_dir = os.path.join(output_base, 'legal')
illegal_dir = os.path.join(output_base, 'illegal')
os.makedirs(legal_dir, exist_ok=True)
os.makedirs(illegal_dir, exist_ok=True)

# 라벨 파일들 루프
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

for label_file in label_files:
    image_file = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)

    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 없음: {image_path}")
        continue

    height, width = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"잘못된 라벨 형식: {line}")
            continue

        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        cropped = img[y1:y2, x1:x2]

        save_dir = legal_dir if int(class_id) == 0 else illegal_dir  # class 0 => legal, class 1 => illegal
        save_name = f"{os.path.splitext(image_file)[0]}_{idx}.jpg"
        save_path = os.path.join(save_dir, save_name)

        cv2.imwrite(save_path, cropped)
        print(f"Saved: {save_path}")