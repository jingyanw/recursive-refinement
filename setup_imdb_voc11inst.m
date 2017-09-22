function imdb = setup_imdb_voc11inst(varargin)
% SETUP_IMDB_VOC11INST: Set up the imdb.

opts.clusterPath = '';
opts = vl_argparse(opts, varargin) ;

% load meta-data
% ------
imdb.classes.name = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',...
               'bus','car', 'cat', 'chair', 'cow',...
               'diningtable', 'dog', 'horse', 'motorbike', 'person',...
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

imdb.imageDir = 'data/VOCdevkit/VOC-SBD/img/' ;
imdb.segPath = strrep(imdb.imageDir, 'img', 'cls-png', '%s.png');
imdb.instPath = strrep(imdb.segPath, 'cls-png', 'inst-png');

imdb.clusters.path = opts.clusterPath;
clusters = load(imdb.clusters.path, 'means', 'assign');
assign = clusters.assign;
clusters = clusters.means;
imdb.clusters.means = clusters;
C = numel(clusters);
nCluster = zeros(1, C);
for c = 1 : C
    nCluster(c) = numel(clusters{c});
end
imdb.clusters.num = nCluster;

% images
% ------
k = 0 ;
for thisSet = {'train', 'val'}
  thisSet = char(thisSet) ;

  fprintf('Loading PASCAL VOC %s set\n', thisSet) ;
  gtids = textread(sprintf('data/VOCdevkit/VOC-SBD/%s.txt',thisSet),'%s');

  k = k + 1 ;
  imdb_.images.name{k} = strcat(gtids,'.jpg');
  N = numel(imdb_.images.name{k});
  
  sizes = zeros(N, 2);
  gtbox = cell(N, 1);
  gtlabel = cell(N, 1);
  gtdist = cell(N, 1);
  gtdistflip = cell(N, 1);

  % Load ground truth objects
  start = tic;

  % for i=1:length(gtids)
  parfor i=1:length(gtids)
    % Read annotation
    [boxes_inst, labels_inst, size_inst, ~, ~, iou_cluster, iou_cluster_flip] = ...
        read_record(imdb, gtids{i}, clusters); % boxes: Mx4

    sizes(i, :) = size_inst;
    gtbox{i} = boxes_inst;
    gtlabel{i} = labels_inst;
    gtdist{i} = iou_cluster;
    gtdistflip{i} = iou_cluster_flip;

    assert(numel(labels_inst) == size(iou_cluster, 1));
    assert(isequal(size(iou_cluster), size(iou_cluster_flip)));

    if mod(i-1, 100) == 0, fprintf('[%s %.1f sec] %d/%d.\n', thisSet, toc(start), i, length(gtids)); end
  end
  
  imdb_.images.set{k}  = k * ones(1, N);
  imdb_.images.size{k} = sizes;
  imdb_.boxes.gtbox{k} = gtbox;
  imdb_.boxes.gtlabel{k} = gtlabel;
  imdb_.boxes.gtdist{k} = gtdist;
  imdb_.boxes.gtdistflip{k} = gtdistflip;
end

imdb.images.name = vertcat(imdb_.images.name{:}) ;
imdb.images.size = vertcat(imdb_.images.size{:}) ;
imdb.images.set  = horzcat(imdb_.images.set{:}) ;
imdb.boxes.gtbox = vertcat(imdb_.boxes.gtbox{:}) ;
imdb.boxes.gtlabel = vertcat(imdb_.boxes.gtlabel{:}) ;
imdb.boxes.gtdist = vertcat(imdb_.boxes.gtdist{:});
imdb.boxes.gtdistflip = vertcat(imdb_.boxes.gtdistflip{:});

% flip
% ------
imdb.boxes.flip = zeros(size(imdb.images.name));

% Add flipped
train = (imdb.images.set == 1) ;
imdb.images.name = vertcat(imdb.images.name, imdb.images.name(train)) ;
imdb.images.set  = horzcat(imdb.images.set, imdb.images.set(train)) ;
imdb.images.size  = vertcat(imdb.images.size, imdb.images.size(train,:)) ;

imdb.boxes.flip = vertcat(imdb.boxes.flip, ones(sum(train),1)) ; % [non-flip, flip]
imdb.boxes.gtbox = vertcat(imdb.boxes.gtbox , imdb.boxes.gtbox(train)) ;
imdb.boxes.gtlabel = vertcat(imdb.boxes.gtlabel, imdb.boxes.gtlabel(train)) ;
imdb.boxes.gtdist = vertcat(imdb.boxes.gtdist, imdb.boxes.gtdistflip(train));
imdb.boxes = rmfield(imdb.boxes, 'gtdistflip');
% when flipped, the sublabel has changed

for i=1:numel(imdb.boxes.gtbox)
  if imdb.boxes.flip(i)
    width = imdb.images.size(i, 1);
    gtbox = imdb.boxes.gtbox{i} ;

    assert(all(gtbox(:,1)<=width));
    assert(all(gtbox(:,3)<=width));

    gtbox(:,1) = width - gtbox(:,3) + 1;
    gtbox(:,3) = width - imdb.boxes.gtbox{i}(:,1) + 1;
    imdb.boxes.gtbox{i} = gtbox;
  end
end
    
% gtsublabel: [N, 1]
imdb.boxes.gtsublabel = cell(size(imdb.boxes.gtlabel));
% train: find sub-label according to cluster
train = find(imdb.images.set == 1);
offset = ones(1, C);
for i = 1 : numel(train)
    idx = train(i);
    gtlabel = imdb.boxes.gtlabel{idx};
    M = numel(gtlabel);
    gtsublabel = zeros(M, 1);
    for j = 1 : M
        cls = gtlabel(j);
        gtsublabel(j) = assign{cls}(offset(cls))';
        offset(cls) = offset(cls) + 1;
    end
    imdb.boxes.gtsublabel{idx} = gtsublabel;
end

for c = 1 : C
    assert(offset(c) == numel(assign{c}) + 1);
end

% val: find sublabels according to max iou
val = find(imdb.images.set == 2);
gtsublabels = cell(1, numel(val));
% for i = 1 : numel(val)
parfor i = 1 : numel(val)
   idx = val(i);
   [~, ~, ~, sublabel, ~, ~, ~] = read_record(imdb, imdb.images.name{idx}(1:end-4), clusters);
   gtsublabels{i} = sublabel;

   if mod(i-1, 100) == 0, fprintf('[val %.1f sec] %d/%d.\n', toc(start), i, numel(val)); end
end

% parfor necessity
for i = 1 : numel(val)
    idx = val(i);
    imdb.boxes.gtsublabel{idx} = gtsublabels{i};
end
