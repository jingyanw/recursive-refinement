function [rec,prec,ap] = VOCevaldet_11inst(dets, VOCopts,cls,draw)
% VOCopts:
%   annocachepath
%   testset ('val')
%   imgsetpath
%   annopath
%   minoverlap

if nargin <= 3
    draw = false;
end

% load test set
cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
if exist(cp,'file')
    load(cp,'gtids','recs');
else
    gtids = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s');
    for i=1:length(gtids)

        % read annotation
        recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
    save(cp,'gtids','recs');
end

% hash image ids
hash=VOChash_init(gtids);

% extract ground truth objects

npos=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % extract objects of class
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    gt(i).BB=cat(1,recs(i).objects(clsinds).bbox)';
    gt(i).diff=[recs(i).objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

% load results
ids = dets.names;
confidence = dets.conf;
BB = dets.bbs';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % find ground truth image
    i=VOChash_lookup(hash,ids{d});
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
		gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
end
