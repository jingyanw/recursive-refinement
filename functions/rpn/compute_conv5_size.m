function [h, w] = compute_conv5_size(h, w)
for i = 1 : 4
    h = floor((h + 1) / 2); % pad 1
    w = floor((w + 1) / 2); % pad 1
end
