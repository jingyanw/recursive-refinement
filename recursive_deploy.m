function net = recursive_deploy(net, varargin)
opts.confThresh = -Inf;
[opts, varargin] = vl_argparse(opts, varargin);

for l = numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') || ...
      isa(net.layers(l).block, 'dagnn.DropOut')
    layer = net.layers(l);
    net.removeLayer(layer.name);
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

% re-organize the Proposal layer
net.setLayerOutputs('proposal', {'rois', 'probrpn'});
net.vars(net.getVarIndex('probrpn')).precious = true;

lProposal2 = net.getLayerIndex('proposal2');
C = numel(net.layers(lProposal2).block.nSubclass);
net.layers(lProposal2).block.confThresh = opts.confThresh;

for cls = 1 : C
    c = @(s) append_c(s, cls);
    
    % class-stage probabilities
    net.renameVar(c('label'), c('probcls'));

    % subclass stage probability
    net.addLayer(c('probsubcls'), dagnn.SoftMax(), c('predcls'), c('probsubcls'));
    net.vars(net.getVarIndex(c('probsubcls'))).precious = true ;

    % subclass proposals
    idxBox = net.getLayerIndex(c('predbbox'));
    net.vars(net.getVarIndex(c('rois'))).precious = true;
    net.vars(net.getVarIndex(c('predbbox'))).precious = true;

    % un-normalize subclass-stage regressor
    blayer = net.layers(idxBox);
    filters = net.params(net.getParamIndex(blayer.params{1})).value;
    biases = net.params(net.getParamIndex(blayer.params{2})).value ;
    
    bboxMean2 = single(net.layers(lProposal2).block.bboxMean2); % [1, 4]
    bboxStd2 = single(net.layers(lProposal2).block.bboxStd2); % [1, 4]
    nSubclass = net.layers(lProposal2).block.nSubclass;

    bboxStd2_f = reshape(bboxStd2, [1, 1, 1, 4]);
    bboxMean2_b = bboxMean2';
    bboxStd2_b = bboxStd2';
    net.params(net.getParamIndex(blayer.params{1})).value = bsxfun(@times, filters, bboxStd2_f);

    biases = biases .* bboxStd2_b;
    net.params(net.getParamIndex(blayer.params{2})).value = bsxfun(@plus, biases, bboxMean2_b);
end

net.mode = 'test' ;
net.rebuild();
