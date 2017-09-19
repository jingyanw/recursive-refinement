function imdb = carve_minival(imdb)
% CARVE_MINIVAL: a helper function to keep a smaller validation set (minival) when evaluating each epoch of the model during training

NMinival = 500;
rng(0);
val = find(imdb.images.set == 2);
NVal = numel(val);
minival = val(randsample(NVal, NMinival));

select = (imdb.images.set == 1);
select(minival) = true;

imdb.images.name = imdb.images.name(select);
imdb.images.size = imdb.images.size(select, :);
imdb.images.set = imdb.images.set(select);

fields = fieldnames(imdb.boxes)';
for f = field
    f = char(f);
    imdb.boxes.(f) = imdb.boxes.(f)(select);
end

fprintf('Carved minival.\n');
