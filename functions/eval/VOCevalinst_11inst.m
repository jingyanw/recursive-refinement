function VOCevalinst(results, minoverlaps, GT)
% VOCEVALINST_11INST: Evaluate instance segmentation on PASCAL VOC11-inst split.
%   Inputs:
%     RESULTS: Predictions
%        .name: name of image
%        .objMap: instance segmentation mask
%        .cls: a list of cls idxs
%        .conf: a list of conf values
%     MINOVERLAPS: Threshold for AP evaluation.
%     GT: Ground-truth.

classes = {'plane', 'bike', 'bird', 'boat', 'bottle', ...
           'bus', 'car', 'cat', 'chair', 'cow', ...
           'table', 'dog', 'horse', 'mbike', 'person', ...
           'plant', 'sheep', 'sofa', 'train', 'tv'};
C = numel(classes);

for minoverlap = minoverlaps
    APs = zeros(1, C);
    fprintf('ovlp: %.1f\n', minoverlap);
    for i = 1 : C
        fprintf('%s: ', classes{i});
        stats = VOCevalinst_ap(results, i, minoverlap, GT);
    
        APs(i) = stats.ap;
        fprintf('AP: %.2f%% | ', stats.ap * 100);
    end
    
    % print results
    fprintf('==============================\n\n');
    fprintf('minoverlap: %.1f\n', minoverlap);
    % first half
    fprintf('|%7s', classes{1:10});
    fprintf('\n');
    fprintf('|%6.2f%%', APs(1:10) * 100);
    fprintf('\n\n');
    % second half
    fprintf('|%7s', classes{11:20});
    fprintf('\n');
    fprintf('|%6.2f%%', APs(11:20) * 100);
    fprintf('\n\n');
    
    fprintf('mAP: %.2f%%\n', mean(APs) * 100);
end

% ------
function stats = VOCevalinst_ap(results, clsidx, minoverlap, GT)
%   Inputs:
%     RESULTS: Predictions.
%        .name: name of image
%        .objMap: instance segmentation mask
%        .cls: a list of cls idxs
%        .conf: a list of conf values
%     MINOVERLAP: Threshold for AP evaluation.
%     GT: Ground-truth.

N = numel(results.names);

% extract ground truth objects
npos=0;
gt(length(N))=struct('det', [], 'objidx', []);

% extract objects of class
for i = 1 : N
    clsinds = find(GT.cls{i} == clsidx);
    gt(i).objidx = clsinds;
    gt(i).det = false(length(clsinds),1);
    npos = npos + numel(clsinds);
end

% load results
if numel(results.names) ~= numel(GT.names)
    warning('Incomplete val results.');
end

% flatten results
pred.imgidx = [];
pred.objidx = [];
pred.conf = [];

% extract objects of class
for i = 1 : N
    clsinds = find(results.cls{i} == clsidx);
    pred.imgidx = [pred.imgidx, i * ones(1, numel(clsinds))];
    pred.objidx = [pred.objidx, clsinds];

    confs = results.conf{i};
    pred.conf = [pred.conf, confs(clsinds)];
end

% sort detections by decreasing confidence
[~, si]=sort(-pred.conf);
pred.imgidx = pred.imgidx(si);
pred.objidx = pred.objidx(si);
pred.conf = pred.conf(si);

% assign detections to ground truth objects
nd=length(pred.conf);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;

for d = 1:nd
    % find ground truth image
    i = pred.imgidx(d);
    name = results.names{i};

    % assign detection to ground truth object if any
    mask = results.objMap{i}{pred.objidx(d)};
    offset = results.offset{i}(pred.objidx(d), :);
    ovmax= -inf;

    objIdxs = gt(i).objidx;
    
    for j = 1 : numel(objIdxs)
        objidx = objIdxs(j);
        maskgt = (GT.objMap{i} == objidx);
        areagt = GT.area{i}(objidx);

        % compute iou
        ov = IOU_mask_with_offset(mask, offset, maskgt, areagt);
        
        if ov>ovmax
            ovmax=ov;
            jmax=j;
        end
    end

    % assign detection as true positive/don't care/false positive
    if ovmax >= minoverlap
       if ~gt(i).det(jmax)
           tp(d)=1;            % true positive
           gt(i).det(jmax)=true;
       else
           fp(d)=1;            % false positive (multiple detection)
       end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

stats.fp = fp;
stats.tp = tp;
stats.npos = npos;
stats.rec = rec;
stats.prec = prec;
stats.ap = ap;

function iou = IOU_mask_with_offset(mask, offset, maskgt, areagt)
% IOU_MASK_WITH_OFFSET: Compute pixel-wise IOU between MASK and MASK_GT.
%
%   Inputs:
%     MASK: tightest size (with offset)
%     MASK_GT: full size (without offset)
left = offset(1);
top = offset(2);

[h, w] = size(mask);
right = left + w - 1;
bottom = top + h - 1;
intersect = mask & maskgt(top:bottom, left:right);
intersect = sum(intersect(:));
iou = intersect / (sum(mask(:)) + areagt - intersect);
