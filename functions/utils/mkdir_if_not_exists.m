function mkdir_if_not_exists(d)
if ~exist(d, 'dir')
    mkdir(d);
end
