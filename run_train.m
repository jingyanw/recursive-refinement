gpus = [4];
imdbPath25 = 'data/imdb-voc11inst-shape-thresh0.25.mat';
imdbPath20 = 'data/imdb-voc11inst-shape-thresh0.20.mat';

% clusters
clusterPath25 = 'data/clusters/clusters-gtbox-thresh0.25.mat';% 1142
clusterPath20 = 'data/clusters/clusters-gtbox-thresh0.20.mat'; % 2951

derOutputs = {'loss_rpn_cls', 1, 'loss_rpn_reg', 1, 'losscls', 1, 'lossbbox', 1};

derOutputs20 = derOutputs;
derOutputs20Normalize = derOutputs;
for i = 1 : 20
    c = @(s) append_c(s, i);
    derOutputs20(end + 1 : end + 4) = {c('losscls'), 1, c('lossbbox'), 1};
    derOutputs20Normalize(end + 1 : end + 4) = {c('losscls'), 1/20, c('lossbbox'), 1/20};
end

baseLR3 = 1/3;
baseLR2 = 1/2;

% train VOC 2011-inst
% recursive_train('expDir', 'models/full-faster-rcnn-again', 'imdbPath', imdbPath25, 'clusterPath', clusterPath25, 'train', struct('gpus', gpus), 'derOutputs', derOutputs, 'baseLR', baseLR2);

recursive_train('expDir', 'models/full-shape25-bkg', 'imdbPath', imdbPath25, 'clusterPath', clusterPath25, 'train', struct('gpus', gpus), 'derOutputs', derOutputs20Normalize, 'baseLR', baseLR3);
