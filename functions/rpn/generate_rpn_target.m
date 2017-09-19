function [labels, targets, weights] = full_generate_anchor_target(varargin)
opts.DEBUG = false;
[opts, varargin] = vl_argparse(opts, varargin);

% required
opts.anchors = []; % [A, 4]
opts.gtboxes = []; % [G, 4]
opts.imsize = [0, 0]; % [H, W]

% optional
opts.nmsThresh = 0.7;
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.feat_stride = 16;
opts.npos = 128;
opts.nneg = 128;
opts.bboxMean = zeros(1, 4);
% opts.bboxStd = [0.1, 0.1, 0.2, 0.2];
opts.bboxStd = ones(1, 4);

[opts, varargin] = vl_argparse(opts, varargin);
if isempty(opts.anchors)
    opts.anchors = generate_anchors();
end
img_h = opts.imsize(1); img_w = opts.imsize(2);
[H, W] = compute_conv5_size(img_h, img_w);

A = size(opts.anchors, 1);
G = size(opts.gtboxes, 1);

% list all boxes
shift_x = (0 : W - 1) * opts.feat_stride;
shift_y = (0 : H -1) * opts.feat_stride;
[shift_x, shift_y] = meshgrid(shift_x, shift_y);
shifts = cat(4, shift_x, shift_y, shift_x, shift_y);
anchors = reshape(opts.anchors, [1, 1, A, 4]);

all_anchors = bsxfun(@plus, anchors, shifts);

inds_inside = all_anchors(:, :, :, 1) >= 0 & ...
              all_anchors(:, :, :, 2) >= 0 & ...
              all_anchors(:, :, :, 3) <= img_w & ...
              all_anchors(:, :, :, 4) <= img_h;

inds_inside_idx = find(inds_inside);

if opts.DEBUG
    fprintf('total_anchors %d\n', numel(inds_inside));
    fprintf('inds_inside %d\n', sum(inds_inside(:)));
end

I = size(inds_inside_idx, 1);
inside_anchors = reshape(all_anchors, [], 4);
inside_anchors = inside_anchors(inds_inside_idx, :);

ovlp = compute_target_iou(all_anchors, opts.gtboxes); % [H, W, A, G]
ovlp = reshape(ovlp, [], G);
ovlp_inside = ovlp(inds_inside_idx, :);

[best_ovlp, best_gt] = max(ovlp_inside, [], 2); % best gt for each anchor [I, 1]
[max_ovlp, best_anchor] = max(ovlp_inside, [], 1); % best anchor for each gt [1, G]
inside_gtboxes = opts.gtboxes(best_gt, :);

inside_targets = bbox_transform(inside_anchors, inside_gtboxes); % [I, 4]
% normalize
inside_targets = bsxfun(@minus, inside_targets, opts.bboxMean);
inside_targets = bsxfun(@rdivide, inside_targets, opts.bboxStd);

labels = zeros(H, W, A, 'single');

labels(inds_inside_idx(best_ovlp <= opts.negThresh)) = -1;
% labels(inds_inside_idx(best_anchor)) = 1; % best anchor for each gt (anchors missed if same ovlp value)
best_anchors = bsxfun(@eq, ovlp_inside, max_ovlp); % [I, G]
best_anchors = any(best_anchors, 2);
labels(inds_inside_idx(best_anchors)) = 1;
% fprintf('#best_anchor: %d\n', numel(best_anchor));
% fprintf('#best_anchors: %d\n', sum(best_anchors));
labels(inds_inside_idx(best_ovlp > opts.posThresh)) = 1; % IOU > 0.7

% subsample
np = sum(labels(:) == 1);
nn = sum(labels(:) == -1);

if np > opts.npos
    idxs_pos = find(labels == 1);
    idxs_ignore = datasample(idxs_pos, np - opts.npos, 'Replace', false);
    labels(idxs_ignore) = 0;
end

batchsize = opts.npos + opts.nneg;
opts.nneg = max(opts.nneg, batchsize - min(opts.npos, np));
if nn > opts.nneg
    idxs_neg = find(labels == -1);
    idxs_ignore = datasample(idxs_neg, nn - opts.nneg, 'Replace', false);
    labels(idxs_ignore) = 0;
end

% reshape regressor target
targets = zeros(H * W * A, 4, 'single');
targets(inds_inside_idx, :) = inside_targets;
% order: [A1_dx, A2_dx, A3_dx, ..., A1_dy, ..., A1_dw, ... A9_dh]
targets = reshape(targets, [H, W, A * 4]);

% order: [A1_dx, A1_dy, A1_dw, A1_dh, A2_dx, ..., A9_dh]
targets = target_transform(targets);

weights = (labels == 1);
weights = single(repelem(weights, 1, 1, 4)); % [A1_dx, A1_dy, A1_dw, A1_dh, ...]
% weights = single(repmat(weights, 1, 1, 4)); % [A1_dx, A2_dx, A3_dx, ...]

if opts.DEBUG
    ts = targets(weights == 1);
    fprintf('RPN mean: %.3f\n', mean(ts));
    fprintf('RPN std: %.3f\n', std(ts));
    keyboard;
end
