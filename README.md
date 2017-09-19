# The more you look, the more you see: towards general object understanding through recursive refinement

This is the source code for training and evaluating the hierarchical detection model with recursive refinement module.

## Installation
1. Clone the repository:
    ```git clone --recursive https://github.com/jingyanw/recursive-refinement.git
    ```

2. Build the NMS module (same as [Faster-RCNN](https://github.com/ShaoqingRen/faster_rcnn/)): run `build_nms.m`


### Downloads
Run `bash download.sh` to download the following data:
    ```
    $ROOT/data/
        pretrained/imagenet-vgg-verydeep-16.mat: pre-trained ImageNet VGG-16 model
        clusters/TODO
    ```


- Model:

