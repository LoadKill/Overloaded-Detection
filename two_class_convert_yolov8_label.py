import os
import json


json_label_dir = r'one-class-dataset\labels\val'
yolo_label_dir = r'YOLOv8_twoClass_format_data\labels\val'

os.makedirs(yolo_label_dir, exist_ok=True)

for filename in os.listdir(json_label_dir):
    if not filename.endswith('.json'):
        continue

    json_path = os.path.join(json_label_dir, filename)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    file_info = data['FILE'][0]

    # 절대좌표 변환 위환 width, height값 추출(from JSON 파일)
    resolution = file_info['RESOLUTION']
    width, height = map(int, resolution.split('*'))

    yolo_lines = []

    for item in file_info['ITEMS']:
        box_str = item['BOX']
        if not box_str:
            continue

        # 클래스 ID 값. (정상차량: 0, 불법차량: 1)
        package = item.get('PACKAGE', '').strip()
        if package == '정상차량':
            class_id = 0
        elif package == '불법차량':
            class_id = 1
        else:
            continue  

        # 절대좌표 변환
        x, y, w, h = map(float, box_str.split(','))
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(yolo_line)

    # YOLO 포맷에 맞게 txt 형식으로 변환
    base = os.path.splitext(filename)[0]
    txt_path = os.path.join(yolo_label_dir, base + '.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))

    print(f"변환 완료: {base}")