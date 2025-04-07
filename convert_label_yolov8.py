import json
import os


CLASS_MAP = {
    "대형차_불법차량": 0,
    "대형차_정상차량": 1,
    "중형차_불법차량": 2,
    "중형차_정상차량": 3,
    "소형차_불법차량": 4,
    "소형차_정상차량": 5
}

json_folder = "/content/rere_augmented/labels"
output_label_folder = "/content/re_convert_test"

def parse_resolution(res_str):
    w, h = res_str.split('*')
    return int(w), int(h)

def convert_json_to_yolo(json_path, output_folder):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON 형식 오류: {json_path}")
        print(f"   -> 오류 내용: {e}")
        return

    for file_data in data.get('FILE', []):
        image_name = file_data.get('FILE_NAME')
        res = file_data.get('RESOLUTION')

        try:
            img_w, img_h = parse_resolution(res)
        except Exception as e:
            print(f"해상도 파싱 실패: {json_path} - {e}")  # 상대좌표 사용 위한 width, height값 필요
            continue

        yolo_lines = []

        for item in file_data.get('ITEMS', []):
            class_name = f"{item.get('SEGMENT', '')}_{item.get('PACKAGE', '')}"
            if class_name not in CLASS_MAP:
                continue
            class_id = CLASS_MAP[class_name]

            box_str = item.get('BOX', '')
            try:
                x, y, w, h = map(float, box_str.split(','))
            except Exception as e:
                print(f"BOX 파싱 오류: {json_path}")
                print(f"   -> BOX 값: '{box_str}'")
                print(f"   -> 오류 내용: {e}")
                continue  # 해당 파일 건너뛰고 다음으로

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            yolo_lines.append(yolo_line)

        # 비어 있는 경우 건너뛰기
        if not yolo_lines:
            continue

        label_filename = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(output_folder, label_filename)

        with open(label_path, "w", encoding='utf-8') as out_file:
            out_file.write("\n".join(yolo_lines))

for root, dirs, files in os.walk(json_folder):
    for file in files:
        if file.endswith(".json"):
            json_path = os.path.join(root, file)
            convert_json_to_yolo(json_path, output_label_folder)