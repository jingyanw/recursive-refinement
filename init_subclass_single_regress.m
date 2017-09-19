function net = init_subclass_single_regress(net, varargin)
% separate linear classifiers built on shared fc7 features

opts.category = 0; % array of integers
opts.nShape = 0; % array of integers: 1xC of all the categories
opts = vl_argparse(opts, varargin);

% CATEGORY: category idx
nCls = opts.nShape + 1;

for cat = opts.category
    c = @(s) append_c(s, cat);
    % roi
    pRelu5 = find(arrayfun(@(a) strcmp(a.name, 'relu5_3'), net.layers)==1);
    net.addLayer(c('roipool'), dagnn.ROIPooling('method','max','transform',1/16,...
      'subdivisions',[7,7],'flatten',0), ...
      {net.layers(pRelu5).outputs{1}, c('rois')}, c('xRP'));
    
    % fc6
    fc6 = dagnn.Conv('size', [7, 7, 512, 4096], 'hasBias', true);
    net.addLayer(c('fc6'), fc6, c('xRP'), c('fc6'), {'fc6f_subclass', 'fc6b_subclass'})
    
    net.addLayer(c('relu6'), dagnn.ReLU(), c('fc6'), c('relu6'));
    net.addLayer(c('drop6'), dagnn.DropOut('rate', 0.5), c('relu6'), c('drop6'))
    
    % fc7
    fc7 = dagnn.Conv('size', [1, 1, 4096, 4096], 'hasBias', true);
    net.addLayer(c('fc7'), fc7, c('drop6'), c('fc7'), {'fc7f_subclass', 'fc7b_subclass'})
    
    net.addLayer(c('relu7'), dagnn.ReLU(), c('fc7'), c('relu7'));
    net.addLayer(c('drop7'), dagnn.DropOut('rate', 0.5), c('relu7'), c('drop7'))
    
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

% init shared fc6/7 parameters
fc6f = net.params(net.getParamIndex('fc6f')).value;
fc6b = net.params(net.getParamIndex('fc6b')).value;
net = initialize_param(net, 'fc6f_subclass', fc6f, 1, 1);
net = initialize_param(net, 'fc6b_subclass', fc6b, 2, 0);

fc7f = net.params(net.getParamIndex('fc7f')).value;
fc7b = net.params(net.getParamIndex('fc7b')).value;
net = initialize_param(net, 'fc7f_subclass', fc7f, 1, 1);
net = initialize_param(net, 'fc7b_subclass', fc7b, 2, 0);
