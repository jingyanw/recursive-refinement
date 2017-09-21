dbstop if error;
addpath bin;
addpath data/devkit/VOCdevkit/VOCcode;
addpath(genpath('functions'));

run matconvnet/matlab/vl_setupnn;
addpath matconvnet/examples/fast_rcnn;
addpath matconvnet/examples/fast_rcnn/bbox_functions;
