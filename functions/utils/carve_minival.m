function imdb = carve_minival(imdb)
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

field = {'gtbox', 'gtlabel', 'gtshape', 'flip', 'pbox', 'plabel', 'piou', 'pgtidx', 'ptarget', 'gtshapeiou', 'pshape', 'gtsublabel', 'gtdist'};
for f = field
    f = char(f);
    if isfield(imdb.boxes, f)
        imdb.boxes.(f) = imdb.boxes.(f)(select);
    end
end

fprintf('Carved minival.\n');
