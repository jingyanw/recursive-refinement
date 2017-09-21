% RUN_TEST: script to test the model
gpus = [2];

expDir = 'models/final-once-more';

recursive_test('expDir', expDir, 'gpu', gpus, 'top1', 300, 'top2', 100, 'conf_subcls', true);
