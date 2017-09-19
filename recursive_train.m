function [net, info] = recursive_train(varargin)
opts.DEBUG = false;
opts.dataDir   = '/data/jingyanw/dataset/pascal/inst/' ;
opts.expDir    = 'models/trash';
opts.imdbPath  = fullfile('imdb/imdb-voc11inst.mat');
opts.modelPath = '/data/jingyanw/pretrained/imagenet-vgg-verydeep-16.mat';
opts.clusterPath = '/home/jingyanw/work/exemplar-pascal/analyze/clusters-gtbox.mat';
opts.derOutputs = {};
opts.rpnPos = 128;
opts.rpnNeg = 128;
opts.classPos = 128;
opts.classNeg = 128;
opts.subclassPos = 128;
opts.subclassNeg = 128;

opts.category = 1:20;
opts.bgThreshLo = 0; % 0.1: hard-mining
opts.keep_neg_n = 500; % after RPN
opts.keep_neg_n_subclass = 300; % after class
opts.baseLR = 1;

opts.multilabel = false;
opts.randomize = false;
opts.randomizeThreshPos = 0.7;
% reduce to multi-class
opts.mlHiThresh  = +Inf;
opts.mlLoThresh = +Inf;
opts.mlWeight = 1;
opts.singleRegress = true;
opts.randomSeed = 0;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.confThresh = -Inf;

opts.train.gpus = [] ;
opts.train.batchSize = 1 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.prefetch = false ; % does not help for single-image batches
opts.train.learningRate = 1e-3 / 256 * [ones(1,5) 0.1*ones(1,2)];
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 12 ;
opts.train.derOutputs = opts.derOutputs;
opts.train.randomSeed = opts.randomSeed;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if ~exist(opts.expDir,'dir')
  mkdir(opts.expDir);
end
save(fullfile(opts.expDir, 'opts'), '-struct', 'opts');

if exist(opts.imdbPath)
    fprintf('Loading imdb...\n');
    imdb = load(opts.imdbPath);
else
    fprintf('Creating imdb...\n');
    imdb = setup_imdb_voc11inst('clusterPath', opts.clusterPath);
    fprintf('Saving imdb...\n');
    if ~exist(opts.imdbPath, 'dir')
        mkdir(fileparts(opts.imdbPath));
    end
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
end
fprintf('done.\n');

scratchDir = '/scratch/jingyanw/dataset/pascal/inst/img/';
if exist(scratchDir, 'dir')
    imdb.imageDir = scratchDir;
end

% use minival
imdb = carve_minival(imdb);

% generate multi-label
% if opts.multilabel
%     imdb = create_multilabel(imdb, 'hiThresh', opts.mlHiThresh, 'loThresh', opts.mlLoThresh);
% end
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = recursive_init('modelPath',opts.modelPath, 'nShape', imdb.clusters.num, 'confThresh', opts.confThresh, ...
  'subclassPos', opts.subclassPos, 'subclassNeg', opts.subclassNeg, 'category', opts.category, 'bgThreshLo', opts.bgThreshLo, 'keep_neg_n', opts.keep_neg_n, 'keep_neg_n_subclass', opts.keep_neg_n_subclass, 'singleRegress', opts.singleRegress, 'DEBUG', opts.DEBUG, 'baseLR', opts.baseLR, 'classPos', opts.classPos, 'classNeg', opts.classNeg);

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% minibatch options
bopts = net.meta.normalization;
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.maxScale = 1000;
bopts.scale = 600;
bopts.bgLabel = numel(imdb.classes.name)+1;
bopts.visualize = 0;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.prefetch = opts.train.prefetch;
bopts.mode = 'train';
bopts.rpnPos = opts.rpnPos;
bopts.rpnNeg = opts.rpnNeg;
bopts.randomize = opts.randomize;
bopts.randomizeThreshPos = opts.randomizeThreshPos;

anchors = generate_anchors();

% warmup  here
%{
iter_warmup = 100;% 100 or 500 doesn't work
warmup = opts.train;
warmup.learningRate = opts.train.learningRate(1) / 10;
warmup.numEpochs = 1;
imdb_warmup = imdb;
set_warmup = zeros(1, numel(imdb_warmup.images.set));
set_warmup(randsample(find(imdb_warmup.images.set == 1), iter_warmup)) = 1;
imdb_warmup.images.set = set_warmup;

[net,info] = cnn_train_dag(net, imdb_warmup, @(i,b) getBatch(bopts,anchors, i,b), warmup);

movefile(fullfile(opts.expDir, 'net-epoch-1.mat'), fullfile(opts.expDir, 'net-epoch-0.mat'));
%}

[net,info] = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,anchors, i,b), ...
                           opts.train) ;

% test
full_test_shape('imdbPath', opts.imdbPath, 'expDir', opts.expDir, 'gpu', opts.train.gpus, 'clusterPath', opts.clusterPath, 'reg2', true, 'singleRegress', opts.singleRegress);
fprintf('Done.\n');

% --------------------------------------------------------------------
function inputs = getBatch(opts, anchors, imdb, batch)
% --------------------------------------------------------------------
if isempty(batch)
  return;
end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im, gtboxes] = recursive_get_batch_single(images,imdb, batch, opts);

if opts.prefetch, return; end

% RPN loss sampling
H = size(im, 1); W = size(im, 2);
[labels, targets, instance_weights] = generate_rpn_target('anchors', anchors, 'gtboxes', gtboxes(1:4, :)', 'imsize', [H, W], 'npos', opts.rpnPos, 'nneg', opts.rpnNeg);

if opts.useGpu > 0
  im = gpuArray(im) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
  % TODO: does it make sense to make GTBOXES GpuArray?
end

inputs = {'input', im, 'rpn_labels', labels, 'rpn_targets', targets, 'rpn_instance_weights', instance_weights, 'imsize', [H, W], 'gtboxes', gtboxes} ;
