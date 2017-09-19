dbstop if error;
addpath bin;
addpath(genpath('functions'));

run matconvnet/matlab/vl_setupnn;
addpath matconvnet/examples/fast_rcnn;
addpath matconvnet/examples/fast_rcnn/bbox_functions;
