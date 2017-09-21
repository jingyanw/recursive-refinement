function recursive_test(varargin)
% RECURSIVE_TEST: Test the model on detection and instance segmentation.

opts.expDir = '';
opts.clusterPath = 'data/clusters/clusters-shape-thresh25.mat';
opts.imdbPath = 'data/imdb-voc11inst-shape-thresh0.25.mat';
opts.epoch = 0;
opts.top1 = 300;
opts.top2 = 100;
opts.confThresh = 0.01;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.epoch == 0
    opts.epoch = findLastCheckpoint(opts.expDir); % find last epoch
end
opts.modelPath = fullfile(opts.expDir, sprintf('net-epoch-%d.mat', opts.epoch));

opts.gpu = [] ;
opts.numFetchThreads = 1 ;
opts.nmsThresh = 0.3 ;
opts.maxPerImage = 100 ;

% ablations
opts.conf_subcls = true;
[opts, varargin] = vl_argparse(opts, varargin) ;

display(opts) ;

if ~isempty(opts.gpu)
  gpuDevice(opts.gpu)
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = load(opts.modelPath, 'net'); % {net, state, stats}
net = dagnn.DagNN.loadobj(net.net) ;

net = recursive_deploy(net, 'confThresh', opts.confThresh, 'top1', opts.top1, 'top2', opts.top2);

if ~isempty(opts.gpu)
  net.move('gpu') ;
end

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
fprintf('Loading precomputed imdb...\n');
imdb = load(opts.imdbPath) ;
fprintf('done.\n');

clusterPath = imdb.clusters.path;
clusters = load(clusterPath);

bopts.averageImage = net.meta.normalization.averageImage;
bopts.useGpu = numel(opts.gpu) >  0 ;
bopts.maxScale = 1000;
bopts.visualize = 0;
bopts.scale = 600;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = opts.numFetchThreads;
bopts.mode = 'test';

% -------------------------------------------------------------------------
%                                                                  Evaluate
% -------------------------------------------------------------------------
VOCinit;
VOCopts.testset='val';

% category classifier
testIdx = find(imdb.images.set == 2) ;
% testIdx = testIdx(1:2000);% TODO delete
NVal = numel(testIdx);

% shape classifier -- find categories
nCls = numel(VOCopts.classes);
probClsVars = nan(1, nCls);
probSubclsVars = nan(1, nCls);
proposalVar = net.getVarIndex('rois_sub');
splitVar = net.getVarIndex('split_sub');
deltaVars = nan(1, nCls);
for cls = 1 : nCls
    c = @(s) append_c(s, cls);
 
    probClsVars(cls) = net.getVarIndex(c('probcls'));
    probSubclsVars(cls) = net.getVarIndex(c('probsubcls'));
    deltaVars(cls) = net.getVarIndex(c('predbbox'));
end

% init -- det
boxscores_nms = cell(nCls, NVal) ;

% inst -- inst
inst.names = imdb.images.name(testIdx);
inst.cls = cell(1, NVal);
inst.conf = cell(1, NVal);
inst.offset = cell(1, NVal);
inst.objMap = cell(1, NVal);

start = tic ;
for t = 1: numel(testIdx)
  batch = testIdx(t);
  inputs = getBatch(bopts, imdb, batch);

  net.eval(inputs) ;
  
  im_size = imdb.images.size(testIdx(t),[2 1]);
  im_h = im_size(1);
  im_w = im_size(2);

  pred_boxes = squeeze(gather(net.vars(proposalVar).value))'; % proposals
  split = gather(net.vars(splitVar).value);
  pred_boxes = mat2cell(pred_boxes, split, 5);
  for cls = 1 : nCls
    pred_box = pred_boxes{cls};
    probs_cls = net.vars(probClsVars(cls)).value; % cls prob
    probs_subcls = squeeze(gather(net.vars(probSubclsVars(cls)).value)); % subcls prob
    box_deltas = squeeze(gather(net.vars(deltaVars(cls)).value));

    % regress from scaled box
    if isempty(pred_box), continue; end
    % fprintf('#: %d\n', size(pred_box, 1));
    pred_box = pred_box(:, 2:5);
 
    if opts.conf_subcls
        conf =  probs_cls .* (1 - probs_subcls(end, :)); % joint
    else
        conf = probs_cls;
    end
    % TODO: use the actual joint subclass probs
 
    [~, subcls_idx] = max(probs_subcls(1:end - 1, :), [], 1);
    if cls == 1, keyboard; end
    factor = max(bopts.scale / im_h, bopts.scale / im_w);
    if any([im_h*factor, im_w*factor] > bopts.maxScale)
      factor = min(bopts.maxScale / im_h, bopts.maxScale / im_w);
    end
  
    factor = 1 / factor;

    bboxes = bbox_scale(pred_box, factor, [im_w, im_h]);
    bboxes = bbox_clip(round(bboxes), im_size); 
  
    boxscore = [bboxes conf'];

    % threshold on the final/joint probability
    % conf_select = (conf > opts.confThresh);
    % boxscore = boxscore(conf_select, :);
    % subcls_idx = subcls_idx(conf_select);
    % conf = conf(conf_select);
    
    % [~,si] = sort(boxscore(:,5),'descend');
    % boxscore = boxscore(si,:);
    % boxscore = boxscore(1:min(size(boxscore,1),opts.maxPerImage),:);
    % shape_idx = shape_idx(si);
    % shape_delta = shape_deltas(:, si);

    M = numel(conf);
    for m = 1 : M
       box = boxscore(m, 1:4);
       delta = box_deltas(:, m)';

       box = bbox_transform_inv(box, delta);
       box = bbox_clip(round(box), im_size);
       boxscore(m, 1:4) = box;
    end

    % NMS
    pick = bbox_nms(double(boxscore),opts.nmsThresh);
    boxscores_nms{cls, t} = boxscore(pick,:) ;
    subcls_idx = subcls_idx(pick);
    probs_subcls = probs_subcls(:, pick);
    conf = conf(pick);

    % write inst
    M = numel(pick);
    inst.cls{t} = horzcat(inst.cls{t}, cls * ones(1, M));
    inst.conf{t} = horzcat(inst.conf{t}, conf);
    objMap = cell(1, M);

    for m = 1 : M
      box = boxscores_nms{cls, t}(m, 1:4);
      left = box(1); right = box(3);
      top = box(2); bottom = box(4);

      w = right - left + 1;
      h = bottom - top + 1;

      exemplar = imresize(clusters.means{cls}{subcls_idx(m)}, [h, w], 'nearest');
      objMap{m} = logical(exemplar);
    end

    offset = boxscores_nms{cls, t}(:, 1:2);

    % M x 2
    inst.offset{t} = vertcat(inst.offset{t}, offset);
    inst.objMap{t} = horzcat(inst.objMap{t}, objMap);

    if false
      idx = boxscores_nms{cls, t}(:, 5) > 0.5;
      if sum(idx)==0, continue; end
      
      figure(1) ; clf ;
      bbox_draw(imread(fullfile(imdb.imageDir,imdb.images.name{testIdx(t)})), ...
                boxscores_nms{cls, t}(idx,:)) ;
      title(['img #' num2str(t) ': ' net.meta.classes.name{cls}]) ;

      drawnow ;
      pause;
    end
  end

  if mod(t-1, 100) == 0, fprintf('[%.1f sec] %d/%d (%.1f HZ)\n', toc(start), t, numel(testIdx), t / toc(start)) ; end
end

gpuDevice([]);

%% PASCAL VOC evaluation
VOCdevkitPath = fullfile('data/devkit', 'VOCdevkit');

% fix voc folders
VOCopts.imgsetpath = '/data/jingyanw/dataset/pascal/inst/%s.txt';
VOCopts.annopath   = '/data/jingyanw/dataset/pascal/voc11/Annotations/%s.xml';
VOCopts.localdir   = fullfile(VOCdevkitPath, 'local','inst');
VOCopts.annocachepath = fullfile(VOCopts.localdir, '%s_anno.mat');

mkdir_if_not_exists(VOCopts.localdir);

% inst eval
% ---
gtPath = 'data/voc11-inst/gt_inst_val.mat';
if ~exist(gtPath)
    setup_inst_gt(imdb);
end
gt = load(gtPath);
VOCevalinst_11inst(inst, [0.5, 0.7], gt);

% det eval
% ---
dets.names = cell(1, 0);
dets.bbs = zeros(0, 4);
dets.conf = zeros(1, 0);
dets.cls = zeros(1, 0);
for i = 1 : NVal
    name = inst.names{i}(1:end-4);
    for c = 1 : nCls
        boxes = boxscores_nms{c, i};
        if isempty(boxes), continue; end
        dets.bbs = [dets.bbs; boxes(:, 1:4)];
        dets.conf = [dets.conf, boxes(:, 5)'];
        
        M = numel(boxes(:, 5));
        dets.names = horzcat(dets.names, repmat({name}, [1, M]));
        dets.cls = horzcat(dets.cls, ones(1, M) * c);
    end
end

tic;
for ovlp = [0.5, 0.7]
    VOCopts.minoverlap = ovlp;
    aps = zeros(1, numel(nCls));
    for c = 1 : nCls
        det.names = dets.names(dets.cls == c);
        det.bbs = dets.bbs(dets.cls == c, :);
        det.conf = dets.conf(dets.cls == c);
        [rec,prec,ap] = VOCevaldet_11inst(det, VOCopts, net.meta.classes.name{c});
        fprintf('[det (%.1f)] %s AP %.1f%%\n', ovlp, VOCopts.classes{c}, 100*ap);
        aps(c) = ap;
    end
    fprintf('mAP (%.1f): %.1f%%\n', ovlp, 100 * mean(aps));
end

fprintf('Done.\n'); keyboard;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if isempty(batch), return; end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,gtboxes] = recursive_get_batch_single(images, imdb, batch, opts);

if opts.useGpu > 0
  im = gpuArray(im) ;
end

% layer is not run if any input is empty, so gtboxes can't be empty
H = size(im, 1);
W = size(im, 2);
inputs = {'input', im, 'gtboxes', nan, 'imsize', [H, W]} ;
