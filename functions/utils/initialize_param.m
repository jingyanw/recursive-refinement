function net = initialize_param(net, name, value, learningRate, weightDecay)
% INITIALIZE_PARAM: Initialize parameters to the network

f = net.getParamIndex(name);
net.params(f).value = single(value);
net.params(f).learningRate = learningRate;
net.params(f).weightDecay = weightDecay;
