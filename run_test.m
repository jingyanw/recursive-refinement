% RUN_TEST: script to test the model

gpus = [1];
modelPath = 'data/models/shape-thresh25-vgg16-epoch7.mat';

recursive_test('modelPath', modelPath, 'gpu', gpus);
