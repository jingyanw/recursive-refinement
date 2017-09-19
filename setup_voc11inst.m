function imdb = setup_voc11inst(varargin)
% SETUP_VOC11INST Setup PASCAL VOC 2011-inst data

opts.instAnno = true; % true: inst-derived boxes, false: orig det boxes
[opts, varargin] = vl_argparse(opts, varargin);

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

imdb.classes.name ={'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',...
               'bus','car', 'cat', 'chair', 'cow',...
               'diningtable', 'dog', 'horse', 'motorbike', 'person',...
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

imdb.imageDir = '/data/jingyanw/dataset/pascal/inst/img/' ;
imdb.segPath = strrep(imdb.imageDir, 'img', 'cls-png');
imdb.segPath = fullfile(imdb.segPath, '%s.png');
imdb.instPath = strrep(imdb.segPath, 'cls-png', 'inst-png');

if ~opts.instAnno
    VOCinit;
    VOCopts.annopath = '/data/jingyanw/dataset/pascal/voc11/Annotations/%s.xml';
end

% -------------------------------------------------------------------------
%                                                                    Images
% -------------------------------------------------------------------------

k = 0 ;
for thisSet = {'train', 'val'}
  thisSet = char(thisSet) ;

  fprintf('Loading PASCAL VOC %s set\n', thisSet) ;
  [gtids,t]=textread(sprintf('/data/jingyanw/dataset/pascal/inst/%s.txt',thisSet),'%s %d');

  k = k + 1 ;
  imdb_.images.name{k} = strcat(gtids,'.jpg');
  imdb_.images.set{k}  = k * ones(size(imdb_.images.name{k}));

  % parfor init
  imdb_images_size = zeros(numel(imdb_.images.name{k}),2);
  imdb_boxes_gtbox = cell(size(imdb_.images.name{k}));
  imdb_boxes_gtlabel = cell(size(imdb_.images.name{k}));

  % Load ground truth objects
  start = tic;
  for i=1:length(gtids)
    % Read annotation
    if opts.instAnno
        [boxes_inst, labels_inst, size_inst] = read_box_from_mask(imdb, gtids{i}); % boxes: Mx4
    else
        rec = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
        boxes_inst = vertcat(rec.objects(:).bbox);
        diff = vertcat(rec.objects(:).difficult);
        [~, labels_inst] = ismember({rec.objects(:).class}, VOCopts.classes);
        size_inst = rec.imgsize(1:2);

        if ~isempty(boxes_inst)
            boxes_inst = boxes_inst(~diff, :);
            labels_inst = labels_inst(~diff)';
        end
    end

    imdb_images_size(i,:) = size_inst;

    % extract objects of class
    BB = boxes_inst;
    label = labels_inst; 

    if ~isempty(BB)
      imdb_boxes_gtbox{i} = BB;
      imdb_boxes_gtlabel{i} = label;
    end
    if mod(i-1, 500) == 0, fprintf('[%s][%.1f sec] %d/%d.\n', thisSet, toc(start), i, length(gtids)); end
  end

  imdb_.images.size{k} = imdb_images_size;
  imdb_.boxes.gtbox{k} = imdb_boxes_gtbox;
  imdb_.boxes.gtlabel{k} = imdb_boxes_gtlabel;
end

imdb.images.name = vertcat(imdb_.images.name{:}) ;
imdb.images.size = vertcat(imdb_.images.size{:}) ;
imdb.images.set  = vertcat(imdb_.images.set{:}) ;
imdb.boxes.gtbox = vertcat(imdb_.boxes.gtbox{:}) ;
imdb.boxes.gtlabel = vertcat(imdb_.boxes.gtlabel{:}) ;

% -------------------------------------------------------------------------
%                                                                   Flipped
% -------------------------------------------------------------------------
imdb.boxes.flip = zeros(size(imdb.images.name));

% Add flipped
trainval = (imdb.images.set == 1) ;
imdb.images.name = vertcat(imdb.images.name, imdb.images.name(trainval)) ;
imdb.images.set  = vertcat(imdb.images.set, imdb.images.set(trainval)) ;
imdb.images.size  = vertcat(imdb.images.size, imdb.images.size(trainval,:)) ;

imdb.boxes.gtbox = vertcat(imdb.boxes.gtbox , imdb.boxes.gtbox(trainval)) ;
imdb.boxes.gtlabel = vertcat(imdb.boxes.gtlabel, imdb.boxes.gtlabel(trainval)) ;
% when flipped, the shape label has changed
imdb.boxes.flip = vertcat(imdb.boxes.flip, ones(sum(trainval),1)) ;

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
