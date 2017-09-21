% RUN_TEST: script to test the model
gpus = [4];

expDir = 'models/full-latest-format';

recursive_test('expDir', expDir, 'gpu', gpus, 'top1', 300, 'top2', 100, 'conf_subcls', true);
