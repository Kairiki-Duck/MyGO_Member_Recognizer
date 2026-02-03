import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from mygo_train import MyGONet


def run_inference(test_folder_path="./try_everything"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 路径与文件夹检查
    test_folder = Path(test_folder_path)
    if not test_folder.exists():
        print(f"提示：找不到测试文件夹 {test_folder.absolute()}")
        print("正在创建该文件夹... 请放入图片后重新运行。")
        test_folder.mkdir(parents=True, exist_ok=True)
        return

    # 2. 加载类别映射
    try:
        with open("class_indices.json", "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        classes_count = len(idx_to_class)
    except FileNotFoundError:
        print("错误: 找不到 class_indices.json，请确认该文件在当前目录下。")
        return

    # 3. 初始化并加载模型
    try:
        model = MyGONet(num_classes=classes_count).to(device)
        model.load_state_dict(torch.load("MyGO_members_recognizer.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 4. 预处理 (必须与训练时完全一致)
    test_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. 执行推理
    print(f"\n{'File Name':<20} | {'Top Prediction':<14} | {'Confidence'}")
    print("-" * 55)

    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_image = False

    for img_path in test_folder.iterdir():
        if img_path.suffix.lower() in supported_formats:
            has_image = True
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = test_tf(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]

                # 获取最高概率
                top_prob, top_idx = torch.max(probs, dim=0)
                class_name = idx_to_class[top_idx.item()]

                print(f"{img_path.name[:20]:<20} | {class_name:<14} | {top_prob.item():.2%}")

            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {e}")

    if not has_image:
        print("文件夹内没有找到有效的图片文件。")


if __name__ == "__main__":
    run_inference()