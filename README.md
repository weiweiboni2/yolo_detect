# yolo_detect

YOLO系列模型训练自己的数据集

[![Cuda](https://img.shields.io/badge/CUDA-12.6-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)  [![Python](https://img.shields.io/badge/Python-3.9-%2314354C?logo=python&logoColor=white)](https://www.python.org/downloads/)  ![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-orange.svg)  ![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-blue.svg)  

## 配置环境

``` python
 pip install ultralytics
```

## 数据准备

- [x] 数据处理参考 [labels](https://github.com/weiweiboni2/labels)

- [x] 数据集目录结构：

```text
  .
├── ./DroneVehicle_det
│   ├── ./DroneVehicle_det/labels
│   │   ├── ./DroneVehicle_det/labels/0.txt
│   │   ├── ./DroneVehicle_det/labels/1.txt
│   │   ├── ./DroneVehicle_det/labels/2.txt
│   │   ├── ./DroneVehicle_det/labels/3.txt
│   │   ├── ...
│   ├── ./DroneVehicle_det/images
│   │   ├── ./DroneVehicle_det/images/0.jpg
│   │   ├── ./DroneVehicle_det/images/1.jpg
│   │   ├── ./DroneVehicle_det/images/2.jpg
│   │   ├── ./DroneVehicle_det/images/3.jpg
│   │   ├── ...
│   └── ./DroneVehicle_det/DroneVehicle_det.yaml
```

- [x] 修改数据加载配置文件，例：

    ```python
    path: E:\python_pj\yolov8\YOLOv8-main\data\DroneVehicle_det
    train: E:\python_pj\yolov8\YOLOv8-main\data\DroneVehicle_det\images\train
    val: E:\python_pj\yolov8\YOLOv8-main\data\DroneVehicle_det\images\val
    test: E:\python_pj\yolov8\YOLOv8-main\data\DroneVehicle_det\images\test
    names:
    0: '机动车：'
    #  0: '机动车：'
    #  1: '机动车：'
    #  2: '机动车：'
    #  3: '机动车：'
    #  4: '机动车：'
    nc: 1

    ```

## 模型训练

[参考](https://github.com/serendipityshe/datasetCreation)

- [x] 训练集：提供涵盖多种特征的不同变换的照片，使模型可以学习到特征与目标之间的映射关系。训练集要尽可能大，能涵盖输入和输出的全部范围。
- [x] 验证集：用于验证模型在训练过程中的效果，并用来调整模型超参数（引导训练方向），验证集和训练集应来自相同的数据分布，但必须是训练集之外的数据样本，大小通常为训练数据集的10-30%。
- [x] 测试集：用于评估模型的最终性能,测试集必须是模型训练过程中未曾使用过的数据。用于评估模型的最终性能，测试集必须是模型训练过程中未曾使用过的数据。来自相同的数据分布，其大小也为训练集的10-30%。
- [x] 命令行训练命令：
  
    ```python
    # 视频预测（可把视频路径换成图片路径，预测图片）：
    yolo task=detect mode=predict model=E:\python_pj\yolov8\YOLOv8-main\runs\detect\train2\weights\best.pt source=E:\python_pj\yolov8\YOLOv8-main\data\DJI_20241030112453_0001_T.mp4 show=True
    # 断点续训(windows下workers只能为0)：
    yolo task=detect mode=train model=runs/detect/exp/weights/last.pt data=ultralytics/datasets/mydata.yaml epochs=100 save=True resume=True batch=4 device=0 workers=0
    # 模型训练：
    yolo task=detect mode=train model=E:\python_pj\yolov8\YOLOv8-main\ultralytics\models\v8\yolov8m.yaml     data=E:\python_pj\yolov8\YOLOv8-main\data\DroneVehicle_det\drone.yaml epochs=300  batch=8 device=0 workers=0
    # 验证模型精度（验证集验证）：
    yolo task=detect mode=val model=E:\python_pj\yolov8\YOLOv8-main\runs\obb\train5\weights\best.pt data=E:\python_pj\yolov8\YOLOv8-main\data\data_enhance\UBW.yaml workers=0

    ```

- [x] 训练代码：
[代码参考](https://blog.csdn.net/weixin_41171614/article/details/136922875?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov8%E6%A8%A1%E5%9E%8B%E9%AA%8C%E8%AF%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-136922875.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187)

    ```python
    # 训练代码，设置超参数：
    train.py
    # 推理：
    detect.py
    # 验证：
    # split :写val就是看验证集的指标，写 test 就是看测试集的指(前提是划分了测试集)
    # batch ：测试速度时一般都设置成1（验证比较稳定），设置越大，速度越快
    val.py
    ```

## 模型输出

- [x] 无人机红外行人检测：
![示例图片](0000164_01068_d_0000162000.png)
  
- [x] 无人机红外车辆检测：
![示例图片](0000164_01068_d_0000162000.png)

- [x] 无人机违建检测：
  ![示例图片](0000164_01068_d_0000162000.png)

## 引用

>@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {<https://github.com/ultralytics/ultralytics>},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
>
