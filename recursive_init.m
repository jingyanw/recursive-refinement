function net = recursive_init(varargin)
opts.debug = false;

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

[opts, varargin] = vl_argparse(opts, varargin) ;
display(opts) ;

% start building subclass stage
nFg = opts.nCls - 1;
drops2 = cell(1, nFg);
targets2 = cell(1, nFg);
labels2 = cell(1, nFg);
instance_weights2 = cell(1, nFg);
for i = 1 : nFg
    drops2{i} = append_c('drop7', i);
    targets2{i} = append_c('targets', i);
    labels2{i} = append_c('label', i);
    instance_weights2{i} = append_c('instance_weights', i);
end

assert(numel(opts.nShape) == opts.nCls - 1);

% output 21 category-specific proposals
net.addLayer('probcls', dagnn.SoftMax(), 'predcls', 'probcls', {});
net.addLayer('proposal2', dagnn.Proposal2(...
        'nSubclass', opts.nShape, 'confThresh', opts.confThresh, ...
        'subclassPos', opts.subclassPos, 'subclassNeg', opts.subclassNeg, ...
        'bgThreshLo', opts.bgThreshLoSubclass, 'keep_neg_n', opts.keep_neg_n_subclass, 'debug', opts.debug), ...
        {'probcls', 'predbbox', 'rois', 'gtboxes', 'imsize'}, ...
        horzcat({'rois_sub', 'split_sub'}, labels2, targets2, instance_weights2));

% add per-category shape classifier
% ======
% roi
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);
net.addLayer('roipool_sub', dagnn.ROIPooling('method','max','transform',1/16,...
  'subdivisions',[7,7],'flatten',0), ...
  {net.layers(pRelu5).outputs{1}, 'rois_sub'}, 'xRP_sub');
% fc6
fc6 = dagnn.Conv('size', [7, 7, 512, 4096], 'hasBias', true);
net.addLayer('fc6_sub', fc6, 'xRP_sub', 'fc6_sub', {'fc6f_subclass', 'fc6b_subclass'});

net.addLayer('relu6_sub', dagnn.ReLU(), 'fc6_sub', 'relu6_sub');
net.addLayer('drop6_sub', dagnn.DropOut('rate', 0.5), 'relu6_sub', 'drop6_sub');

% fc7
fc7 = dagnn.Conv('size', [1, 1, 4096, 4096], 'hasBias', true);
net.addLayer('fc7_sub', fc7, 'drop6_sub', 'fc7_sub', {'fc7f_subclass', 'fc7b_subclass'})

net.addLayer('relu7_sub', dagnn.ReLU(), 'fc7_sub', 'relu7_sub');
net.addLayer('drop7_sub', dagnn.DropOut('rate', 0.5), 'relu7_sub', 'drop7_sub');

% init shared fc6/7 parameters
fc6f = net.params(net.getParamIndex('fc6f')).value;
fc6b = net.params(net.getParamIndex('fc6b')).value;
net = initialize_param(net, 'fc6f_subclass', fc6f, 1, 1);
net = initialize_param(net, 'fc6b_subclass', fc6b, 2, 0);

fc7f = net.params(net.getParamIndex('fc7f')).value;
fc7b = net.params(net.getParamIndex('fc7b')).value;
net = initialize_param(net, 'fc7f_subclass', fc7f, 1, 1);
net = initialize_param(net, 'fc7b_subclass', fc7b, 2, 0);

% split activations for proposals
net.addLayer('split_sub', dagnn.Split2(), {'drop7_sub', 'split_sub'}, drops2);

% add leaf nodes
net = init_subclass_single_regress(net, 'category', opts.category, 'nShape', opts.nShape);

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = net.params(i-1).learningRate * 2;
end

% ------
function net = init_subclass_single_regress(net, varargin)
% separate linear classifiers built on shared fc7 features

opts.category = 0; % array of integers
opts.nShape = 0; % array of integers: 1xC of all the categories
opts = vl_argparse(opts, varargin);

% CATEGORY: category idx
nCls = opts.nShape + 1;

for cat = opts.category
    c = @(s) append_c(s, cat);
    
    % fc8
    fc8f = 0.01 * randn(1, 1, 4096, nCls(cat), 'single');
    fc8b = zeros(1, nCls(cat), 'single');
    
    fc8 = dagnn.Conv('size', [1, 1, 4096, nCls(cat)], 'hasBias', true);
    net.addLayer(c('predcls'), fc8, c('drop7'), c('predcls'), {c('fc8f'), c('fc8b')})
    net = initialize_param(net, c('fc8f'), fc8f, 1, 1);
    net = initialize_param(net, c('fc8b'), fc8b, 2, 0);
    
    % cls
    % softmax loss | multiclass
    net.addLayer(c('losscls'), dagnn.Loss('loss', 'softmaxlog'), {c('predcls'), c('label')}, c('losscls'), {});
    
    % reg (single)
    pf = 0.001 * randn(1, 1, 4096, 4, 'single');
    pb = zeros(1, 4, 'single');
    
    predbbox = dagnn.Conv('size', [1, 1, 4096, 4], 'hasBias', true);
    net.addLayer(c('predbbox'), predbbox, c('drop7'), c('predbbox'), {c('predbboxf'), c('predbboxb')});
    net = initialize_param(net, c('predbboxf'), pf, 1, 1);
    net = initialize_param(net, c('predbboxb'), pb, 2, 0);
    
    % reg -- huber loss
    net.addLayer(c('lossbbox'), dagnn.LossSmoothL1(), {c('predbbox'), c('targets'), c('instance_weights')}, c('lossbbox'), {});
end

