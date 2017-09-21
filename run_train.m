% RUN_TRAIN: Script to train the model.

gpus = [4];
imdbPath25 = 'data/imdb/imdb-voc11inst-shape-thresh25.mat';
% clusters
clusterPath25 = 'data/clusters/clusters-shape-thresh25.mat'; % 1142

derOutputs = {'loss_rpn_cls', 1, 'loss_rpn_reg', 1, 'losscls', 1, 'lossbbox', 1};

derOutputs20 = derOutputs;
for i = 1 : 20
    c = @(s) append_c(s, i);
    derOutputs20(end + 1 : end + 4) = {c('losscls'), 1, c('lossbbox'), 1};
end

baseLR3 = 1/3;

% train VOC 2011-inst
recursive_train('expDir', 'models/final-once-more', 'imdbPath', imdbPath25, 'clusterPath', clusterPath25, 'train', struct('gpus', gpus), 'derOutputs', derOutputs20, 'baseLR', baseLR3);
