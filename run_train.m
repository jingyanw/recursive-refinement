% RUN_TRAIN: Script to train the model.

gpus = [2];
% imdb
imdbPath25 = 'data/imdb/imdb-voc11inst-shape-thresh25.mat';
% clusters: 1142 in total
clusterPath25 = 'data/clusters/clusters-shape-thresh25.mat';

% train VOC 2011-inst
recursive_train('expDir', 'data/models/shape-thresh25-vgg16', 'imdbPath', imdbPath25, 'clusterPath', clusterPath25, 'train', struct('gpus', gpus));
