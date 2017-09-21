function anchors = generate_anchors()
% GENERATE_ANCHORS: generate the 9 anchor boxes as defined in Faster-RCNN.
% https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py

BASE_SIZE = 16;
RATIOS = [0.5, 1, 2];
SCALES = [8, 16, 32];

base_anchor = [1, 1, BASE_SIZE, BASE_SIZE];
% ratio_enum: change aspect ratio
[w, h, x_ctr, y_ctr] = whctrs(base_anchor);
side = sqrt(w * h);

anchors = zeros(0, 4);
for ratio = RATIOS
    ws = round(side / sqrt(ratio));
    hs = round(side * sqrt(ratio));
    for scale = SCALES
        ww = ws * scale;
        hh = hs * scale;
        anchor = [x_ctr, y_ctr, x_ctr, y_ctr] + [-(ww - 1) / 2, -(hh - 1) / 2, (ww - 1) / 2, (hh - 1) / 2];
        anchors = [anchors; anchor];
    end
end

function [w, h, x_ctr, y_ctr] = whctrs(anchor)
left = anchor(1); top = anchor(2); right = anchor(3); bottom = anchor(4);
w = right - left + 1;
h = bottom - top + 1;
x_ctr = (left + right) / 2;
y_ctr = (top + bottom) / 2;
