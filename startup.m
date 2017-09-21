dbstop if error;
addpath bin;
addpath data/devkit/VOCdevkit/VOCcode;
addpath(genpath('functions'));

matconvnetDir = 'matconvnet-recursive';
run(fullfile(matconvnetDir, 'matlab/vl_setupnn'));
addpath(fullfile(matconvnetDir, 'examples/fast_rcnn'));
addpath(fullfile(matconvnetDir, 'examples/fast_rcnn/bbox_functions'));
