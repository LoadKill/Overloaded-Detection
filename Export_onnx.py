import torch
import timm
import torch.nn as nn


# 모델 학습 때랑 동일하게 구성
model = timm.create_model('xception', pretrained=False, num_classes=1)
model.fc = nn.Sequential(nn.Dropout(p=0.3), model.fc)  # 학습 때와 동일하게 Dropout 추가
model.load_state_dict(torch.load('issue_15_stage2_xception_model_epoch70.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # (batch_size, 3채널, height, width)

# ONNX 변환
torch.onnx.export(
    model, 
    dummy_input,
    "xception_issue_15_stage2_epoch70.onnx",   # 출력 파일명
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)