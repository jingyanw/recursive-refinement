dbstop if error;
addpath(genpath('functions'));

if exist('bin', 'dir')
    addpath bin;
else
    fprintf('NMS module is not installed.\n');
end

if exist('data/devkit/VOCdevkit/VOCcode')
    addpath data/devkit/VOCdevkit/VOCcode;
else
    fprintf('VOC devkit is not downloaded.\n');
end

% matconvnet
run matconvnet/matlab/vl_setupnn;
addpath matconvnet/examples/fast_rcnn;
addpath matconvnet/examples/fast_rcnn/bbox_functions;
