function run_demo(varargin)
% RUN_DEMO: Run a trained model on a single image.
opts.gpu = [1] ;

opts.modelPath = 'models/full-latest-format/net-epoch-4.mat'; % TODO: use updated path
opts.clusterPath = 'data/clusters/clusters-shape-thresh25.mat';
opts.top1 = 300;
opts.top2 = 100;
opts.confThresh = 0.9;
[opts, varargin] = vl_argparse(opts, varargin) ;

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
% Network initialization
% -------------------------------------------------------------------------
net = load(opts.modelPath, 'net'); % {net, state, stats}
net = dagnn.DagNN.loadobj(net.net) ;

net = recursive_deploy(net, 'confThresh', opts.confThresh, 'top1', opts.top1, 'top2', opts.top2);

if ~isempty(opts.gpu)
  net.move('gpu') ;
end

% -------------------------------------------------------------------------
% Database initialization
% -------------------------------------------------------------------------
clusters = load(opts.clusterPath);
I = imread('data/demo.jpg');
im = single(bsxfun(@minus, single(I), net.meta.normalization.averageImage));
if ~isempty(opts.gpu),  im = gpuArray(im); end
[H, W, ~] = size(im);
inputs = {'input', im, 'gtboxes', nan, 'imsize', [H, W]};


% -------------------------------------------------------------------------
% Evaluate
% -------------------------------------------------------------------------
% shape classifier -- find categories
nCls = 20;
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

net.eval(inputs) ;

pred_boxes = squeeze(gather(net.vars(proposalVar).value))'; % proposals
split = gather(net.vars(splitVar).value);
pred_boxes = mat2cell(pred_boxes, split, 5);

boxscores_nms = cell(1, nCls);
inst.cls = zeros(1, 0);
inst.conf = zeros(1, 0);
inst.offset = zeros(0, 2);
inst.objMap = cell(0, 1);
for cls = 1 : nCls
  pred_box = pred_boxes{cls};
  probs_cls = net.vars(probClsVars(cls)).value; % cls prob
  probs_subcls = squeeze(gather(net.vars(probSubclsVars(cls)).value)); % subcls prob
  box_deltas = squeeze(gather(net.vars(deltaVars(cls)).value));

  % regress from scaled box
  if isempty(pred_box), continue; end
  bboxes = pred_box(:, 2:5);

  if opts.conf_subcls
      conf =  probs_cls .* (1 - probs_subcls(end, :)); % joint
  else
      conf = probs_cls;
  end
  % TODO: use the actual joint subclass probs

  [~, subcls_idx] = max(probs_subcls(1:end - 1, :), [], 1);
  bboxes = bbox_clip(round(bboxes), [H, W]); 

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
     box = bbox_clip(round(box), [H, W]);
     boxscore(m, 1:4) = box;
  end

  % NMS
  pick = bbox_nms(double(boxscore),opts.nmsThresh);
  boxscores_nms{cls} = boxscore(pick,:) ;
  subcls_idx = subcls_idx(pick);
  probs_subcls = probs_subcls(:, pick);
  conf = conf(pick);

  % write inst
  M = numel(pick);
  inst.cls = horzcat(inst.cls, cls * ones(1, M));
  inst.conf = horzcat(inst.conf, conf);
  objMap = cell(1, M);

  for m = 1 : M
    box = boxscores_nms{cls}(m, 1:4);
    left = box(1); right = box(3);
    top = box(2); bottom = box(4);

    w = right - left + 1;
    h = bottom - top + 1;

    exemplar = imresize(clusters.means{cls}{subcls_idx(m)}, [h, w], 'nearest');
    objMap{m} = logical(exemplar);
  end

  offset = boxscores_nms{cls}(:, 1:2);

  % M x 2
  inst.offset = vertcat(inst.offset, offset);
  inst.objMap = horzcat(inst.objMap, objMap);
end

gpuDevice([]);
  
% visualize
% ------
legends = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',...
           'bus','car', 'cat', 'chair', 'cow',...
           'diningtable', 'dog', 'horse', 'motorbike', 'person',...
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

% det
figure(1) ; clf;
bbox_draw_all(I, boxscores_nms, legends);
title('demo (det)');
drawnow;

% instance segmentation
figure(2); clf;
mask_draw_all(I, inst.offset, inst.objMap);
title('demo (inst)');
drawnow;


fprintf('Done.\n');
