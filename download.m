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

% PASCAL VOC2012 data (using the box annotations for detection evaluation) + devkit
if ~exist('data/VOCdevkit/VOC2012')
    fprintf('Prepare PASCAL-VOC (raw data)...\n');
    untar('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar', 'data');
end

if ~exist('data/VOCdevkit/VOCcode')
    fprintf('Prepare PASCAL-VOC (devkit)...\n');
    untar('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar', 'data');
end

% SBD augmentation
SBDDir = 'data/VOCdevkit/VOC-SBD';
if ~exist(SBDDir)
    fprintf('Prepare PASCAL-VOC (sds augmentation)...\n');
    mkdir_if_not_exists('data/tmp');
    websave('data/tmp/sds.tgz', 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz');
    untar('data/tmp/sds.tgz', 'data/tmp');
    movefile('data/tmp/benchmark_RELEASE/dataset', SBDDir);
    mat2png(SBDDir, 'cls', labelColors(21));
    mat2png(SBDDir, 'inst', labelColors(35));
end

fprintf('Done.\n');

function mat2png(pascalDir, task, cmap)
% DIR: 'cls' or 'inst'
sourceDir = fullfile(pascalDir, task); 
destDir = [sourceDir, '-png'];
mkdir_if_not_exists(destDir);

files = dir(fullfile(sourceDir, '*.mat'));
N = numel(files);
for i = 1 : N
    name = files(i).name;
    field = ['GT' task];
    im = load(fullfile(sourceDir, name), field);
    im = uint8(im.(field).Segmentation);

    imwrite(uint8(im), cmap, fullfile(destDir, [name(1:end-4) '.png']), 'png');
    if mod(i-1, 1000) == 0 || i == N, fprintf('Convert %s image %d/%d.\n', task, i, N); end

end

function cmap = labelColors(N)
% From https://github.com/vlfeat/matconvnet-fcn/blob/master/fcnTest.m
% N: number of colors
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
% N: number of colors
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
