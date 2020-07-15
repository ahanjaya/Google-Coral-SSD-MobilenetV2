# How to use  USB Accelerator Google-Coral 

The main idea of this guide is to train Mobilenet_SSD_V2 on Tensorflow-Object-Detection-API and run on USB Accelerator Google Coral

## Getting Started

These instructions will show you how to run YoloV3 on darknet and pytorch and also provided how to do training on custom object detection


---

## Prerequisites

What things you need to run this codes:

1. CUDA --> [Installation](https://hackmd.io/@ahanjaya/HkioE4SNr)
2. Anaconda --> [Installation](https://www.anaconda.com/products/individual)

Tested on MSI-GP63 (Leopard 8RE):

1. 8th Gen. Intel® Core™ i7 processor
2. GTX 1060 6GB GDDR5 with desktop level performance
3. Memory 16GB DDR4 2666MHz
4. SSD 256 GB
5. Ubuntu 16.04.06 LTS (with ROS Kinetic)


---
## Table of Contents

[TOC]

---

## Anaconda Environment

requirement.txt

```
tensorflow-gpu==1.14
pillow
lxml
Cython
contextlib2
jupyter
matplotlib
pandas
opencv-python
pathlib
gast==0.2.2
imgaug
numpy
tqdm
pycocotools
```

```
1. conda create -n coral_env pip python=3.5
2. conda activate coral_env
3. pip install --upgrade pip
4. conda install -c anaconda protobuf
5. pip install -r requirements.txt
```

```
# From within 1.TensorFlow/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Install EdgeTPU [compiler](https://coral.ai/docs/edgetpu/compiler/#system-requirements)
```
1. curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
2. echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
3. sudo apt-get update
4. sudo apt-get install edgetpu-compiler
```

---

## TensorFlow Object Detection API

### 1. Configure PYTHONPATH environment variable
The Installation docs suggest that you either run, or add to ~/.bashrc file, the following command, which adds these packages to your PYTHONPATH:
```
# From within 1.TensorFlow/research/
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/1.TensorFlow/research/slim
```
### 2. Compile Protobufs
```
1. cd 1.TensorFlow/
2. protoc object_detection/protos/*.proto --python_out=.
3. python setup.py build
4. python setup.py install
```

### 3. TF's COCO evaluation wrapper 

1. addtion in [~/anaconda3/envs/coral_env/lib/python3.5/site-packages/pycocotools/cocoeval.py](https://github.com/pialin/MetricsPerCategoryImplementation/blob/master/tensorflow/research/pycocotools/cocoeval.py):from line 499 to line 648
2. addtion in [1.TensorFlow/research/object_detetcion/metrics/coco_tools.py](https://github.com/pialin/MetricsPerCategoryImplementation/blob/master/tensorflow/research/object_detection/metrics/coco_tools.py):from line 240 to line 244

### 4. Test TensorFlow setup to verify it works
```
1. cd 1.TensorFlow/research
2. python object_detection/builders/model_builder_tf1_test.py
```

---
## Image-Dataset

### 1. Converting Video to Images

### 2. Labeling - [Ybat](https://github.com/drainingsun/ybat)
```
1. cd 2.Images-Dataset/1.Ybat/
2. touch custom_classes.txt # fill the customs classes names
3. **double clicks** : ybat.html
4. **click** Images: browse button # select all images in directory
    example:
    select all images in 2.Images-Dataset/2.Image-BBoxes-Augmentation/images_src
5. **click** Classes: browse button # select all images in directory
    example:
    select 2.Images-Dataset/1.Ybat/custom_classes.txt 
        or 2.Images-Dataset/1.Ybat/arrow.txt 
6. start labeling by using mouse click and next arrow key
7. **click** Save VOC button # when finish labeling
8. extract label *.xml files into --> 2.Images-Dataset/2.Image-BBoxes-Augmentation/images_src
```

### 3. Image Augmentation
**a. Augmentation**
```
cd 2.Images-Dataset/2.Image-BBoxes-Augmentation/scripts
jupyter
Image-BBoxes-Augmentation.ipynb
```
**b. Partition Dataset and Labels**
```
cd 2.Images-Dataset/2.Image-BBoxes-Augmentation/scripts
python partition_images_labels.py ../images_src/ 
```

---
## Training Custom Dataset
### **1. Copy Images and Labels**
```
1. cp -r 2.Images-Dataset/2.Image-BBoxes-Augmentation/train 1.TensorFlow/workspace/training_demo/images
2. cp -r 2.Images-Dataset/2.Image-BBoxes-Augmentation/test 1.TensorFlow/workspace/training_demo/images
3. cp 2.Images-Dataset/2.Image-BBoxes-Augmentation/train_labels.csv 1.TensorFlow/workspace/training_demo/annotations
4. cp 2.Images-Dataset/2.Image-BBoxes-Augmentation/test_labels.csv 1.TensorFlow/workspace/training_demo/annotations
```

### **2. Creating labelmap**
```
1. cd 1.TensorFlow/workspace/training_demo/annotations
2. touch labelmap.pbtxt
```
file inside *labelmap.pbtxt*
```
item {
  id: 1
  name: 'straight'
}

item {
  id: 2
  name: 'left'
}

item {
  id: 3
  name: 'right'
}
```
### **3. Creating TensorFlow Records**

edit this 1.TensorFlow/workspace/scripts/generate_tfrecord.py line 34:43 to be similiar with *labelmap.pbtxt*
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'straight':
        return 1
    elif row_label == 'left':
        return 2
    elif row_label == 'right':
        return 3
    else:
        None
```

```
1. cd 1.TensorFlow/workspace/scripts
2. python generate_tfrecord.py --csv_input=../training_demo/annotations/train_labels.csv --image_dir=../training_demo/images/train --output_path=../training_demo/annotations/train.record
3. python generate_tfrecord.py --csv_input=../training_demo/annotations/test_labels.csv --image_dir=../training_demo/images/test --output_path=../training_demo/annotations/test.record

```

### **4. Configuring a Training Pipeline**
we will use pretrained model of ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
```
1. cd 1.TensorFlow/workspace/training_demo/pre-trained-model/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
2. cp pipeline.config ../../training_history/arrow_pipeline.config
```
edit arrow_pipeline.config
```
Line 3: num_classes: 3
Line 134: batch_size: 16 # based on GPU memory
Line 157: fine_tune_checkpoint: "pre-trained-model/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"
Line 162: label_map_path: "../training_demo/annotations/labelmap.pbtxt"
Line 164: input_path: "../training_demo/annotations/train.record"
Line 174: label_map_path: "../training_demo/annotations/labelmap.pbtxt"
Line 178: input_path: "../training_demo/annotations/test.record"
Line 135-138 *delete these lines, because horizontal flip cause arrow false detection 
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
```

### **5. Download Pre-Trained Model**
[Tensorflow-Zoo (Tf1)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

```
1. cd 1.TensorFlow/workspace/training_demo/pre-trained-model
2. sh download_model.sh
```


### **6. Training the Model**
```
1. cd 1.TensorFlow/workspace/scripts
2. python model_main.py --alsologtostderr --model_dir=../training_demo/training_history/ --pipeline_config_path=../training_demo/training_history/arrow_pipeline.config 
```

## Exporting a Trained Inference Graph
### 1. Export Inference Graph (.ckpt) to Frozen Graph (.pb)
converting Mobilenet+SSD to tflite at some point they use export_tflite_ssd_graph.py

```
python export_tflite_ssd_graph.py --pipeline_config_path=../training_demo/training_history/arrow_pipeline.config --trained_checkpoint_prefix=../training_demo/training_history/model.ckpt-<number-of-latest-checkpoint>  --output_directory=../training_demo/inference_graph/ --add_postprocessing_op=true
```

### 2. Convert Frozen Graph (.pb) to Quantized TfLite (.tflite)
```
python to_tflite.py ../training_demo/inference_graph/tflite_graph.pb
```

### 3. Compile Quantized (.tflite) to a file compatible with the Edge TPU. 
```
edgetpu_compiler ../training_demo/inference_graph/tflite_graph.tflite
```
the result will be "tflite_graph_edgetpu.tflite"

### 4. Copy tflite_graph_edgetpu.tflite
```
1. cd 1.TensorFlow/workspace/training_demo/inference_graph
2. cp tflite_graph_edgetpu.tflite ../../../../3.Coral/1.Inference/all_models/
```


---
## USB Accelerator Google-Coral

### 1. [How to Install](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime)
a. Add our Debian package repository to your system:

```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
```

b. Install the Edge TPU runtime

```
sudo apt-get install libedgetpu1-std
```

c. Now connect the USB Accelerator to your computer using the provided USB 3.0 cable. If you already plugged it in, **remove** it and **replug** it so the newly-installed udev rule can take effect.


d. [Install the TensorFlow Lite Library - Python 3.5](https://www.tensorflow.org/lite/guide/python)

```
sudo pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl
```


### 2. [Run a model using the TensorFlow Lite API](https://coral.ai/docs/accelerator/get-started/#3-run-a-model-using-the-tensorflow-lite-api)
**a. Download the example code from GitHub:**

```
cd 3.Coral
git clone https://github.com/google-coral/tflite.git
```
**b. Download the bird classifier model, labels file, and a bird photo:**

```
cd tflite/python/examples/classification
bash install_requirements.sh
```

**c. Run the image classifier with the bird photo:**

```
python3 classify_image.py \
--model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels models/inat_bird_labels.txt \
--input images/parrot.jpg
```

### 3. [Test your Coral EdgeTPU Installation](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/) 
```
$ python3
>>> import edgetpu
>>> edgetpu.__version__
'2.14.1'
```

#### Note:
1. to find libedgetpu.so --> `/usr/lib/x86_64-linux-gnu$`


---
## Running inference on USB Accelerator

### 1. Default SSD_MobilentV2 on COCO
```
1. cd 3.Coral/1.Inference/
2. sh download_models.sh
3. cd scripts/
4. python detect.py
```
### 2. Run on custom model
edit model and labels config in *detect.py*
```
Line 75: default_model = 'tflite_graph_edgetpu.tflite'
Line 76: default_labels = 'arrow_labels.txt'
```

---

References
---
1. https://github.com/tensorflow/models/tree/master/research/object_detection

---
## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::
