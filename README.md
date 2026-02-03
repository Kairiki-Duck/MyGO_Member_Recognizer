# MyGO!!!!! Member Recognizer 

这是一个基于CNN的轻量级图像识别项目，旨在识别 **MyGO!!!!!** 中的五位成员。

## 项目特性
- **可视化训练**：支持 TensorBoard 实时监控训练曲线。
- **测试文件**：支持批量识别 `try_everything` 文件夹中的图片
- **开箱即用**：Release页面中有已经训练好的模型，可以直接调用

## 项目局限性
- 只能够识别正脸的照片，并且头占图片的比例应该至少有50%。
- 如果对图片进行色彩上的调整，如加黑白滤镜等，会较大地影响预测的准确率。

## 环境要求
在开始之前，请确保你的电脑已安装 Python 3.8+。
一键安装所需依赖：
```bash
pip install -r requirements.txt
```

## 数据集
在Release页面下载数据集，并放在指定的目录中开始训练。文件结构如下：
```text
   data/
   ├── pictures_train/
   │   ├── Anon/
   │   ├── Rana/
   │   ├── Soyo/
   │   ├── Taki/
   │   └── Tomori/
   └── pictures_test/
       ├── Anon/
       ├── Rana/
       ├── Soyo/
       ├── Taki/
       └── Tomori/
```


---

# MyGO!!!!! Member Recognizer

A lightweight CNN-based image recognition project designed to identify the five members of the band **MyGO!!!!!**.

## Features
- **Visualized Training**: Real-time monitoring of training curves via TensorBoard.
- **Batch Inference**: Batch identification of images in the `try_everything` folder.
- **Out-of-the-box**: The Release page contains pre-trained models that can be used directly.

## Limitations
- **Framing**: Best suited for front-facing portraits where the head occupies at least 50% of the image.
- **Filtering**: Significant color adjustments (e.g., black and white filters) may considerably reduce prediction accuracy.

## Requirements
Ensure you have **Python 3.8+** installed.
Install dependencies with one click:
```bash
pip install -r requirements.txt
```

## Dataset
To train the model yourself, download the dataset from the [Releases page] and extract it to the root directory. The structure of dataset folder should be like this:
```text
   data/
   ├── pictures_train/
   │   ├── Anon/
   │   ├── Rana/
   │   ├── Soyo/
   │   ├── Taki/
   │   └── Tomori/
   └── pictures_test/
       ├── Anon/
       ├── Rana/
       ├── Soyo/
       ├── Taki/
       └── Tomori/
```