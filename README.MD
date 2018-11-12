
基于 MobileNet 模型， 使用 Tensorflow 的 Java API 进行图片的分类以及图形内物体识别。

# Overview

## 1. 图像分类

### 模型下载

地址： https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md

可以挑选合适的模型，比如： mobilenet_v2_1.4_224。

### 标签下载地质

https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet_metadata.txt

### 测试图片

https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG

### 代码位置

完成上述文件下载后，将它们放在resources目录下，然后就可以允许 MobileNetClassification 的 Main 函数了。


## 2. 图形物体识别

### 模型下载

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

可以挑选合适的模型，比如： ssd_mobilenet_v2_coco。

### 标签下载地质

https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt

### 测试图片

https://github.com/tensorflow/models/blob/master/research/object_detection/test_images/image2.jpg

### 代码位置

完成上述文件下载后，将它们放在resources目录下，然后就可以允许 MobileNetDetector 的 Main 函数了。

# 运行环境

## CPU 环境

当找不到GPU时，代码会运行在CPU上。

## GPU 环境

### 第一步：准备机器

找个有 GPU 的机器。

### 第二步：构建环境

这里推荐docker。 

dockerfile如下：

    FROM tensorflow/tensorflow:1.11.0-gpu
    WORKDIR /
    RUN apt-get update
    RUN apt-get -y install maven openjdk-8-jdk
    RUN apt-get autoremove
    CMD ["/bin/bash", "-l"]
 
### 第三步： 打包项目代码

首先添加以下 gpu 依赖：

    <dependency>
        <groupId>org.tensorflow</groupId>
        <artifactId>libtensorflow_jni_gpu</artifactId>
        <version>${tensorflow_version}</version>
    </dependency>

然后打包项目就可以了。


