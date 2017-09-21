function mask_draw_all(im, offsets, objMaps, colors)
% MASK_DRAW_ALL: Visualize instance segmentation results.
%
%   Inputs:
%     IM: Image [H, W, 3]
%     OFFSETS: [N, 2]
%     OBJMAPS: cell(1, N)
%     COLORS: [N, 3]

N = numel(objMaps);

if nargin < 4
    colors = rand(N, 3);
end

if isa(im, 'gpuArray')
    im = gather(im);
end

imagesc(im);
axis image;
axis off;
set(gcf, 'Color', 'white');

[H, W, ~] = size(im);
hold on;
for i = 1 : N
    m = false(H, W);
    [h, w] = size(objMaps{i});
    left = offsets(i, 1); top = offsets(i, 2);
    m(top : top + h - 1, left : left + w - 1) = objMaps{i};

    mask = zeros(H, W, 3);
    for c = 1 : 3
        mask(:, :, c) = m* colors(i, c);
    end
    imagesc(mask, 'Alphadata', m * 0.5);
end

hold off;
