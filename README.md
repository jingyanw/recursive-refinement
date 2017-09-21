# The more you look, the more you see: towards general object understanding through recursive refinement

This is the source code for training and evaluating the hierarchical detection model with recursive refinement module.

If you run into any problems using the code, please file an issue, or contact [Jingyan Wang](mailto:jingyanw@cs.cmu.edu).
## Installation
1. Clone the repo:
```Shell
# use the --recursive option to clone the MatConvNet submodule
git clone --recursive https://github.com/jingyanw/recursive-refinement.git
```
2. Build the `matconvnet` directory. Follow the MatConvNet [installation instructions](http://www.vlfeat.org/matconvnet/install/).

3. Build the NMS module (same as [Faster-RCNN](https://github.com/ShaoqingRen/faster_rcnn/)): run `build_nms.m`


## Downloads
Run `download.m` to download the following data to `$ROOT/data`:

- Pre-trained ImageNet VGG-16 model:

 `pretrained/imagenet-vgg-verdeep-16.mat`

- The cluster subclasses with using the IOU metric on the object masks. The IOU between two shape clusters within the same class is roughly 0.75:

 `clusters/clusters-shape-thresh25.mat`
- The devkit for PASCA-VOC:

  `devkit`
- The PSACAL-VOC2011 raw images and annotations with [SBD augmentation](http://home.bharathh.info/pubs/codes/SBD/download.html):

  `voc11-inst`
- A trained VGG-16 model:

  `models/shape-thresh25-vgg16-epoch7.mat`

## Demo
Run `run_demo.m`. The demo performs detection and instance segmentation on an example image.

## Training
Run `run_train.m` to train the model with the default setting. This approximately takes 2 days on a single GPU. Multi-GPU support is untested.

## Testing
Run `run_test.m` to test the provided off-the-shelf model, with the following performance on detection and instance segmentation:

#### Detection
| AP (@0.5) | AP (@0.7) |
|:---:|:---:|
| TODO | TODO |

#### Instance segmentation
| AP^r (@0.5) | AP^r (@0.7) |
|:---:|:---:|
| TODO | TODO |
