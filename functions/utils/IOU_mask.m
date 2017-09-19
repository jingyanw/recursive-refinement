% previously also called IOU_inst, mask_iou
function iou = IOU_mask(mask1, mask2)
assert(isequal(size(mask1), size(mask2)));
mask1 = logical(mask1);
mask2 = logical(mask2);

intersect = (mask1 & mask2);
union = (mask1 | mask2);
iou = sum(intersect(:)) / sum(union(:));
