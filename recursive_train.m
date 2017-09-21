function [net, info] = recursive_train(varargin)
% RECURSIVE_TRAIN: Train the model.

opts.debug = false;
opts.dataDir   = '/data/jingyanw/dataset/pascal/inst/' ;
opts.expDir    = 'models/trash';
opts.imdbPath  = 'data/imdb/imdb-voc11inst-shape-thresh25.mat');
opts.modelPath = 'data/pretrained/imagenet-vgg-verydeep-16.mat';
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
opts.rpn_sigma = 1;
opts.baseLR = 1;

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
opts.train.numEpochs = 7;
opts.train.derOutputs = opts.derOutputs;
opts.train.randomSeed = opts.randomSeed;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Database initialization
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

% TODO: delete following
scratchDir = '/scratch/jingyanw/dataset/pascal/inst/img/';
if exist(scratchDir, 'dir')
    imdb.imageDir = scratchDir;
end

% use minival
imdb = carve_minival(imdb);
% -------------------------------------------------------------------------
% Network initialization
% -------------------------------------------------------------------------
net = recursive_init('modelPath',opts.modelPath, ...
        'nShape', imdb.clusters.num, 'confThresh', opts.confThresh, ...
        'subclassPos', opts.subclassPos, 'subclassNeg', opts.subclassNeg, ...
        'category', opts.category, 'bgThreshLo', opts.bgThreshLo, ...
        'keep_neg_n', opts.keep_neg_n, ...
        'keep_neg_n_subclass', opts.keep_neg_n_subclass, ...
        'baseLR', opts.baseLR, 'rpn_sigma', opts.rpn_sigma, ...
        'classPos', opts.classPos, 'classNeg', opts.classNeg, ...
        'debug', opts.debug);

% --------------------------------------------------------------------
% Train
% --------------------------------------------------------------------
% minibatch options
bopts = net.meta.normalization;
bopts.useGpu = numel(opts.train.gpus) >  0 ;
bopts.maxScale = 1000;
bopts.scale = 600;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.prefetch = opts.train.prefetch;
bopts.mode = 'train';
bopts.rpnPos = opts.rpnPos;
bopts.rpnNeg = opts.rpnNeg;

anchors = generate_anchors();

[net,info] = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,anchors, i,b), ...
                           opts.train) ;

% test
full_test_shape('imdbPath', opts.imdbPath, 'expDir', opts.expDir, 'gpu', opts.train.gpus, 'clusterPath', opts.clusterPath);
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
[labels, targets, instance_weights] = ...
    generate_rpn_target('anchors', anchors, 'gtboxes', gtboxes(1:4, :)', ...
                        'imsize', [H, W], 'npos', opts.rpnPos, 'nneg', opts.rpnNeg);

if opts.useGpu > 0
  im = gpuArray(im) ;
  targets = gpuArray(targets) ;
  instance_weights = gpuArray(instance_weights) ;
  % TODO: does it make sense to make GTBOXES GpuArray?
end

inputs = {'input', im, 'rpn_labels', labels, 'rpn_targets', targets, ...
          'rpn_instance_weights', instance_weights, 'imsize', [H, W], ...
          'gtboxes', gtboxes} ;
