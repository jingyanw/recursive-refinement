% RUN_TEST: script to test the model

gpus = [3];
modelPath = 'models/final-once-more/net-epoch-6.mat';

recursive_test('modelPath', modelPath, 'gpu', gpus);
