function download()
% DOWNLOAD: Download and processnecessary files.

mkdir_if_not_exists('data');

% Pre-trained ImageNet VGG-16 model
pretrainedDir = 'data/pretrained';
if ~exist(pretrainedDir, 'dir')
    fprintf('Prepare pretrained models...');
    mkdir(pretrainedDir);
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', ...
             fullfile(pretrainedDir, 'imagenet-vgg-verydeep-16.mat'));
end

% PASCAL VOC11-inst
pascalDir = 'data/voc11-inst';
if ~exist(pascalDir, 'dir')
    fprintf('Prepare VOC11-inst data...');
    mkdir(pascalDir);
    untar('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz', pascalDir);
    movefile(fullfile(pascalDir, 'benchmark_code_RELEASE/dataset'), fullfile(pascalDir));

end

% PASCAL devkit
devkitDir = 'data/devkit';
fprintf('Prepare VOCdevkit...\n');
untar('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar', 'VOCdevkit.tar', devkitDir);

