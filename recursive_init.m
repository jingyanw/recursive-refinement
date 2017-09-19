function net = recursive_init(varargin)
opts.DEBUG = false;
% OPTS: Faster-RCNN
opts.modelPath = '';
opts.nCls = 21;
opts.bgThreshLo = 0;
opts.classPos = 128;
opts.classNeg = 128;
opts.subclassPos = 128;
opts.subclassNeg = 128;
opts.keep_neg_n = +Inf;
opts.baseLR = 1;
[opts, varargin] = vl_argparse(opts, varargin);

% init Faster-RCNN backbone
net = init_faster_rcnn_backbone(opts);
for p = 1 : net.getParamIndex('conv5_3b') % conv1 - conv5
    net.params(p).learningRate = net.params(p).learningRate * opts.baseLR;
end

% OPTS: subclass stage
opts.nShape = 0;
opts.confThresh = -Inf;
opts.category = [];
opts.bgThreshLoSubclass = 0;
opts.keep_neg_n_subclass = +Inf;

opts.singleRegress = true;
[opts, varargin] = vl_argparse(opts, varargin) ;
display(opts) ;

% start building subclass stage
nFg = opts.nCls - 1;
rois2 = cell(1, nFg);
targets2 = cell(1, nFg);
labels2 = cell(1, nFg);
instance_weights2 = cell(1, nFg);
for i = 1 : nFg
    rois2{i} = append_c('rois', i);
    targets2{i} = append_c('targets', i);
    labels2{i} = ['label_' num2str(i)];
    instance_weights2{i} = ['instance_weights_' num2str(i)];
end

assert(numel(opts.nShape) == opts.nCls - 1);

% output 21 category-specific proposals
net.addLayer('probcls', dagnn.SoftMax(), 'predcls', 'probcls', {});
net.addLayer('proposal2', dagnn.Proposal2(...
        'nSubclass', opts.nShape, 'confThresh', opts.confThresh, ...
        'subclassPos', opts.subclassPos, 'subclassNeg', opts.subclassNeg, ...
        'bgThreshLo', opts.bgThreshLoSubclass, 'keep_neg_n', opts.keep_neg_n_subclass, 'singleRegress', opts.singleRegress, 'DEBUG', opts.DEBUG), ...
        {'probcls', 'predbbox', 'rois', 'gtboxes', 'imsize'}, ...
        horzcat(rois2, labels2, targets2, instance_weights2));

% add per-category shape classifier
% ======
if opts.singleRegress
    net = init_subclass_single_regress(net, 'category', opts.category, 'nShape', opts.nShape);
else
    error('Not implemented.');
end

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = net.params(i-1).learningRate * 2;
end
