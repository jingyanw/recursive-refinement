% RUN_TEST: script to test the model
gpus = [2];

modelPath = 'models/final-once-more/net-epoch-7.mat';

recursive_test('modelPath', modelPath, 'gpu', gpus, 'top1', 300, 'top2', 100, 'conf_subcls', true);
