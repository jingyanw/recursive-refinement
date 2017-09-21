% RUN_TRAIN: Script to train the model.

gpus = [4];
% imdb
imdbPath25 = 'data/imdb/imdb-voc11inst-shape-thresh25.mat';
% clusters: 1142 in total
clusterPath25 = 'data/clusters/clusters-shape-thresh25.mat';

derOutputs20 = {'loss_rpn_cls', 1, 'loss_rpn_reg', 1, 'losscls', 1, 'lossbbox', 1};

for i = 1 : 20
    c = @(s) append_c(s, i);
    derOutputs20(end + 1 : end + 4) = {c('losscls'), 1, c('lossbbox'), 1};
end

baseLR3 = 1/3;

% train VOC 2011-inst
recursive_train('expDir', 'data/models/shape-thresh25-vgg16', 'imdbPath', imdbPath25, 'clusterPath', clusterPath25, 'train', struct('gpus', gpus), 'derOutputs', derOutputs20, 'baseLR', baseLR3);
