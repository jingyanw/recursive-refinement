function bbox_draw_all(im, boxes, legends, colors)
% BBOX_DRAW_ALL: Visualize detection results.
%
% BOXES: cell(1, N), each cell is box [Nc, 4] or [Nc, 5]
% LEGENDS: string cell [1, N]
% COLORS: [N, 3]

N = numel(boxes);

if nargin < 4
    colors = repmat([1, 0, 0], [N, 1]);
end

if isa(im, 'gpuArray')
    im = gather(im);
end

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');
% hold on;

for i = 1 : N
    if isempty(boxes{i})
        continue;
    end

    for j = 1:size(boxes{i})
        box = boxes{i}(j, 1:4);
        if size(boxes{i}, 2) >= 5 % with conf
            score = boxes{i}(j, end);
            linewidth = 2;
            rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors(i, :));
            label = sprintf('%s : %.3f', legends{i}, score);
            text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
        else % without conf
            linewidth = 2;
            rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', colors(i, :));
            label = sprintf('%s(%d)', legends{i}, i);
            text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
        end
    end

end

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)

rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
