function [imo, gts] = full_get_batch_single(images, imdb, batch, opts)
% GTS: [6, G]

% opts.maxScale = 1000;
% opts.scale = 600;
% opts.interpolation = 'bicubic';
% opts.averageImage = [];
% opts.numThreads = 2;
% opts.prefetch = true;
% opts.mode = 'train' OR 'test'

% opts.randomize
% opts.randomizeThreshPos

assert(numel(batch) == 1);

if isempty(images)
  imo = [] ;
  gts = [] ;
  return ;
end

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
  vl_imreadjpeg(images, 'numThreads',opts.numThreads,'prefetch') ;
  imo = [] ;
  gts = [];
  return ;
end

if fetch
  ims = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
else
  ims = images ;
end

% rescale images and rois
imSize = size(ims{1});

h = imSize(1);
w = imSize(2);

factor = max(opts.scale(1)/h,opts.scale(1)/w);

if any([h*factor,w*factor]>opts.maxScale)
  factor = min(opts.maxScale/h,opts.maxScale/w);
end

if abs(factor-1)>1e-3
  imre = imresize(ims{1},factor,'Method',opts.interpolation);
else
  imre = ims{1};
end

if imdb.boxes.flip(batch)
  imre = imre(:,end:-1:1,:);
end

imreSize = size(imre);

switch opts.mode
    case 'train'
        gtbox = imdb.boxes.gtbox{batch};
        gtlabel = imdb.boxes.gtlabel{batch};
        gtsublabel = imdb.boxes.gtsublabel{batch};
        if opts.randomize % randomize gtsublabel
            gtdist = imdb.boxes.gtdist{batch};
            assert(numel(gtsublabel) == size(gtdist, 1));
            for i = 1 : numel(gtsublabel)
                ious = gtdist(i, 1:imdb.clusters.num(gtlabel(i)));
                candidates = (ious > opts.randomizeThreshPos);
                candidates(gtsublabel(i)) = true;
                candidates = find(candidates);
                gtsublabel(i) = datasample(candidates, 1);
            end
        end
        
        % adapt bounding boxes into new coord
        gtbox = bbox_scale(gtbox, factor, [imreSize(2) imreSize(1)]);
        gts = [gtbox'; gtlabel'; gtsublabel'];
    case 'test'
        gts = nan;
end

% subtract mean
if ~isempty(opts.averageImage)
  imre = bsxfun(@minus,imre, opts.averageImage);
end
imo = single(imre);
