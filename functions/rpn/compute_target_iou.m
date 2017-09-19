function ovlp = compute_target_iou(anchors, gtboxes)
% ANCHORS: [H, W, A, 4]
% BOXES: [G, 4]

% OVLP: [H, W, A, G]

G = size(gtboxes, 1);
boxes = reshape(gtboxes, [1, 1, 1, G, 4]);

ovlp_w = bsxfun(@min, anchors(:, :, :, 3), boxes(:, :, :, :, 3)) - bsxfun(@max, anchors(:, :, :, 1), boxes(:, :, :, :, 1)) + 1;
ovlp_h = bsxfun(@min, anchors(:, :, :, 4), boxes(:, :, :, :, 4)) - bsxfun(@max, anchors(:, :, :, 2), boxes(:, :, :, :, 2)) + 1;
ovlp_w = max(ovlp_w, 0); ovlp_h = max(ovlp_h, 0);
area_ovlp = ovlp_w .* ovlp_h; % [H, W, A, G]

w_anchors = anchors(:, :, :, 3) - anchors(:, :, :, 1) + 1;
h_anchors = anchors(:, :, :, 4) - anchors(:, :, :, 2) + 1;
area_anchors = w_anchors .* h_anchors; % [H, W, A]

w_boxes = gtboxes(:, 3) - gtboxes(:, 1) + 1;
h_boxes = gtboxes(:, 4) - gtboxes(:, 2) + 1;
area_boxes = w_boxes .* h_boxes;
area_boxes = reshape(area_boxes, [1, 1, 1, G]);

ovlp = area_ovlp ./ (bsxfun(@plus, area_anchors, area_boxes) - area_ovlp);
