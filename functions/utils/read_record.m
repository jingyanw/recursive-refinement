function [boxes, labels, sizes, shapes, shapes_flip, ious_cluster, ious_cluster_flip] = read_record(imdb, name, clusters)
% NAME: e.g. 2008_000008

clsMap = imread(sprintf(imdb.segPath, name));
instMap = imread(sprintf(imdb.instPath, name));

M = max(instMap(:));
nCluster = imdb.clusters.num;

boxes = zeros(M, 4);
labels = zeros(M, 1);
shapes = zeros(M, 1);
shapes_flip = zeros(M, 1);
if numel(nCluster) == 1
    ious_cluster = zeros(M, nCluster);
    ious_cluster_flip = zeros(M, nCluster);
else
    classes = unique(clsMap(:))';
    classes = classes(classes > 0); % bkg
    nClusterMax = max(nCluster(classes));

    ious_cluster = nan(M, nClusterMax);
    ious_cluster_flip = nan(M, nClusterMax);
end

for j = 1 : M
    mask = (instMap == j);
    cls = unique(clsMap(mask));
    assert(numel(cls) == 1); assert(cls > 0);

    left = min(find(sum(mask)));
    right = max(find(sum(mask)));
    top = min(find(sum(mask, 2)));
    bottom = max(find(sum(mask, 2)));

    boxes(j, :) = [left, top, right, bottom];
    labels(j) = cls;

    m = mask(top : bottom, left: right);
    [cluster_idx, ious] = choose_exemplar(m, clusters{cls});
    [cluster_idx_flip, ious_flip] = choose_exemplar(fliplr(m), clusters{cls});
    C = numel(clusters{cls});

    shapes(j) = cluster_idx;
    shapes_flip(j) = cluster_idx_flip;
    ious_cluster(j, 1:C) = ious;
    ious_cluster_flip(j, 1:C) = ious_flip;
end
sizes = size(clsMap);
sizes = sizes([2 1]);

function [exemplar_idx, ious] = choose_exemplar(m, cluster)
[h, w] = size(m);
C = numel(cluster);

ious = zeros(1, C);
for c = 1 : C
    shape = cluster{c};
    iou = IOU_mask(m, imresize(shape, [h, w], 'nearest'));
    ious(c) = iou;
end

[~, exemplar_idx] = max(ious);
