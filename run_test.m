gpus = [2];

expDir = 'models/full-faster-rcnn-split-edited';

recursive_test('expDir', expDir, 'gpu', gpus, 'top1', 300, 'top2', 100, 'conf_subcls', true);
