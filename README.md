# The more you look, the more you see: towards general object understanding through recursive refinement

This is the source code for training and evaluating the hierarchical detection model with recursive refinement module.

## Installation
1. Clone the repository:
    ```Shell
    git clone --recursive https://github.com/jingyanw/recursive-refinement.git
    ```
2. Build MatConvNet. Follow the [installation instructions](http://www.vlfeat.org/matconvnet/install/).

3. Build the NMS module (same as [Faster-RCNN](https://github.com/ShaoqingRen/faster_rcnn/)): run `build_nms.m`


## Downloads
Run `matlab.m` to download the following data to `$ROOT/data`:
    
- Pre-trained ImageNet VGG-16 model: `pretrained/imagenet-vgg-verdeep-16.mat`
        
- The cluster subclasses with using the IOU metric on the object masks, `clusters/clusters-shape-thresh25.mat`
- The devkit for PASCA-VOC: `devkit`
- The PSACAL-VOC2011 raw images and annotations with [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) augmentation: `voc11-inst`
- A trained VGG-16 model:

## Demo
Run `run_demo.m`. The demo performs detection and instance segmentation on a single image.

## Training
Run `run_train.m`.

## Testing
Run `run_test.m`.
