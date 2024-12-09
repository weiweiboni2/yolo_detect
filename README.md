# yolo_detect

YOLO系列模型训练自己的数据集

[![Cuda](https://img.shields.io/badge/CUDA-12.6-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)  [![Python](https://img.shields.io/badge/Python-3.9-%2314354C?logo=python&logoColor=white)](https://www.python.org/downloads/)  ![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-orange.svg)  ![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-blue.svg)  

``` python
 pip install ultralytics
```

## 数据集准备

- [x] 数据集data目录结构：

```text
  .
├── ./data
│   ├── ./data/labels
│   │   ├── ./data/labels/0.txt
│   │   ├── ./data/labels/1.txt
│   │   ├── ./data/labels/2.txt
│   │   ├── ./data/labels/3.txt
│   │   ├── ...
│   ├── ./data/images
│   │   ├── ./data/images/0.jpg
│   │   ├── ./data/images/1.jpg
│   │   ├── ./data/images/2.jpg
│   │   ├── ./data/images/3.jpg
│   │   ├── ...
│   └── ./data/data.yaml
```

## 模型训练

- [x] 修改配置文件，进行模型训练：

    ```python
    # 运行train.py脚本，对模型进行训练
    python train.py
    ```

- [x] 训练集：
- [x] 验证集：
- [x] 测试集：

## 模型推理

- [x] 模型推理：
[X-anylabeling](https://github.com/CVHub520/X-AnyLabeling)  

- X-anylabeling是一款开源的图像标注工具，支持多种格式的标签文件。
- 将预训练权重转换为onnx格式，导入X-anylabeling模型，进行自动推理，实现半自动标注。

    ```python
    # 安装依赖
    # 运行pt_onnx.py脚本，对权重进行转换
    pip install ultralytics onnx onnxruntime
    python pt_onnx.py
    ```

- 自定义模型的配置文件：

    ```txt
    # type 类型是固定的几个类别(官网查看对应类别)，name是模型名称，display_name是显示名称(找到模型的唯一标识符)
    # 将权重与配置文件放置在同一目录下

    type: yolov8_obb
    name: yolov8m_obb_UBW-r20241206
    display_name: yolov8m_obb_UBW_2
    model_path: E:\python_pj\X-AnyLabeling-main\models\UBW\best.onnx
    classes:
    - UBW
  
    ```

## 模型输出

## 引用

>@ARTICLE{sun2020drone,
  title={Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3168279}
}
>
>@misc{X-AnyLabeling,
  year = {2023},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {Advanced Auto Labeling Solution with Added Features},
  howpublished = <https://github.com/CVHub520/X-AnyLabeling}}>
}
>
