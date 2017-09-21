function setup_inst_gt(imdb)
% SETUP_INST_GT: Set up and save validation ground-truth for the 
%   instance segmentaiton task.

fprintf('Creating inst ground-truth...\n');
val = find(imdb.images.set == 2);
N = numel(val);

gt.area = cell(1, N);
gt.cls = cell(1, N);
gt.names = imdb.images.name(val)';
gt.objMap = cell(1, N);

t = tic;
for i = 1 : N
    idx = val(i);

    name = imdb.images.name{idx}(1:end-4);
    clsMap = imread(sprintf(imdb.segPath, name));
    objMap = imread(sprintf(imdb.instPath, name));

    gt.objMap{i} = objMap;

    M = max(objMap(:));
    area = zeros(1, M);
    cls = zeros(1, M);
    for m = 1 : M
        mask = (objMap == m);
        area(m) = sum(mask(:));

        c = unique(clsMap(mask));
        assert(numel(c) == 1); assert(c > 0);
        cls(m) = c;
    end
    gt.area{i} = area;
    gt.cls{i} = cls;
    if mod(i-1, 500) == 0, fprintf('[%s %.1f sec] %d/%d.\n', mfilename, toc(t), i, N); end
end

save('data/voc11-inst/gt_inst_val.mat', '-struct', 'gt');
