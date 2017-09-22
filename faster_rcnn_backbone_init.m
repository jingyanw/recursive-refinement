function net = faster_rcnn_backbone_init(varargin)
% FASTER_RCNN_BACKBONE_INIT: initialize the Faster-RCNN model

opts.debug = false;
opts.modelPath = 'data/pretrained/imagenet-vgg-verydeep-16.mat';
opts.nCls = 21;
opts.bgThreshLo = 0;
opts.classPos = 128;
opts.classNeg = 128;
opts.bboxStd = ones(1, 4);
opts.rpn_sigma = 1;
opts.keep_neg_n = +Inf;
[opts, varargin] = vl_argparse(opts, varargin) ;
display(opts) ;

% Load the pretrained ImageNet VGG-16 model.
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);

% Add drop-out layers.
relu6p = find(cellfun(@(a) strcmp(a.name, 'relu6'), net.layers)==1);
relu7p = find(cellfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);

drop6 = struct('type', 'dropout', 'rate', 0.5, 'name','drop6');
drop7 = struct('type', 'dropout', 'rate', 0.5, 'name','drop7');
net.layers = [net.layers(1:relu6p) drop6 net.layers(relu6p+1:relu7p) drop7 net.layers(relu7p+1:end)];

% Change loss for FC layers.
fc8p = find(cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1);
net.layers{fc8p}.name = 'predcls';
net.layers{fc8p}.weights{1} = 0.01 * randn(1,1,size(net.layers{fc8p}.weights{1},3),opts.nCls,'single');
net.layers{fc8p}.weights{2} = zeros(1, opts.nCls, 'single');

% Skip pool5.
pPool5 = find(cellfun(@(a) strcmp(a.name, 'pool5'), net.layers)==1);
net.layers = net.layers([1:pPool5-1,pPool5+1:end-1]);

% Convert to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

%% RPN
pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);

% conv
rpn_convf = 0.01 * randn(3, 3, 512, 512, 'single');
rpn_convb = zeros(1, 512, 'single');
conv_rpn = dagnn.Conv('size', [3, 3, 512, 512], 'hasBias', true, 'pad', [1, 1, 1, 1]);
net.addLayer('rpn_conv', conv_rpn, net.layers(pRelu5).outputs{1}, 'rpn_conv', {'rpn_convf', 'rpn_convb'});
net = initialize_param(net, 'rpn_convf', rpn_convf, 1, 1);
net = initialize_param(net, 'rpn_convb', rpn_convb, 2, 0);

% relu
net.addLayer('rpn_relu', dagnn.ReLU(), 'rpn_conv', 'rpn_relu');

% rpn-score
rpn_scoref = 0.01 * randn(1, 1, 512, 9, 'single'); % 9 (anchor)
rpn_scoreb = zeros(1, 9, 'single');
score_rpn = dagnn.Conv('size', [1, 1, 512, 9], 'hasBias', true);
net.addLayer('rpn_score', score_rpn, 'rpn_relu', 'rpn_score', {'rpn_scoref', 'rpn_scoreb'});
net = initialize_param(net, 'rpn_scoref', rpn_scoref, 1, 1);
net = initialize_param(net, 'rpn_scoreb', rpn_scoreb, 2, 0);

% rpn-regress
rpn_regf = 0.001 * randn(1, 1, 512, 9 * 4, 'single');
rpn_regb = zeros(1, 36, 'single');
reg_rpn = dagnn.Conv('size', [1, 1, 512, 36], 'hasBias', true);
net.addLayer('rpn_reg', reg_rpn, 'rpn_relu', 'rpn_reg', {'rpn_regf', 'rpn_regb'});
net = initialize_param(net, 'rpn_regf', rpn_regf, 1, 1);
net = initialize_param(net, 'rpn_regb', rpn_regb, 2, 0);

% compute target
% rpn-score loss
net.addLayer('loss_rpn_cls', dagnn.Loss('loss', 'logistic'), ...
    {'rpn_score', 'rpn_labels'}, 'loss_rpn_cls');
% rpn-regress loss
net.addLayer('loss_rpn_reg', dagnn.LossSmoothL1('sigma', opts.rpn_sigma), ...
    {'rpn_reg', 'rpn_targets', 'rpn_instance_weights'}, 'loss_rpn_reg');

% proposal
net.addLayer('rpn_prob', dagnn.Sigmoid(), 'rpn_score', 'rpn_prob');
net.addLayer('proposal', dagnn.Proposal('bgThreshLo', opts.bgThreshLo, ...
        'classPos', opts.classPos, 'classNeg', opts.classNeg, ...
        'bboxStd', opts.bboxStd, 'keep_neg_n', opts.keep_neg_n, ...
        'debug', opts.debug), ...
    {'rpn_prob', 'rpn_reg', 'gtboxes', 'imsize'}, ...
    {'rois', 'label', 'targets', 'instance_weights'});

% Add ROIPooling layer.
pFc6 = (arrayfun(@(a) strcmp(a.name, 'fc6'), net.layers)==1);
net.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
  'subdivisions',[7,7],'flatten',0), ...
  {net.layers(pRelu5).outputs{1},'rois'}, 'xRP');

pRP = (arrayfun(@(a) strcmp(a.name, 'roipool'), net.layers)==1);
net.layers(pFc6).inputs{1} = net.layers(pRP).outputs{1};

% Add softmax loss layer.
pFc8 = (arrayfun(@(a) strcmp(a.name, 'predcls'), net.layers)==1);
net.renameVar(net.layers(pFc8).outputs{1}, 'predcls');
net.addLayer('losscls',dagnn.Loss(), ...
  {'predcls','label'}, 'losscls',{});

% Add bbox regression layer.
pparFc8 = (arrayfun(@(a) strcmp(a.name, 'predclsf'), net.params)==1);
pdrop7 = (arrayfun(@(a) strcmp(a.name, 'drop7'), net.layers)==1);
net.addLayer('predbbox',dagnn.Conv('size',[1 1 size(net.params(pparFc8).value,3) 84],'hasBias', true), ...
  net.layers(pdrop7).outputs{1},'predbbox',{'predbboxf','predbboxb'});

net.params(end-1).value = ...
    0.001 * randn(1,1,size(net.params(pparFc8).value,3),84,'single');
net.params(end).value = zeros(1,84,'single');

net.addLayer('lossbbox',dagnn.LossSmoothL1(), ...
  {'predbbox','targets','instance_weights'}, ...
  'lossbbox',{});

net.rebuild();

% No decay for bias and set learning rate to 2
for i=2:2:numel(net.params)
  net.params(i).weightDecay = 0;
  net.params(i).learningRate = net.params(i-1).learningRate * 2;
end

% freeze conv1 and conv2
for i = 1 : 8
    pname = net.params(i).name;
    assert(strcmp(pname(1:5), 'conv1') || strcmp(pname(1:5), 'conv2'));
    net.params(i).weightDecay = 0;
    net.params(i).learningRate = 0;
end

% Change image-mean as in fast-rcnn code
net.meta.normalization.averageImage = ...
  reshape([122.7717 102.9801 115.9465],[1 1 3]);

net.meta.normalization.interpolation = 'bilinear';
net.meta.classes.name = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', ...
    'bus', 'car', 'cat', 'chair', 'cow', ...
    'diningtable', 'dog', 'horse', 'motorbike', 'person', ...
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', ...
    'background' };
