classdef Proposal2 < dagnn.Layer
  
  properties
    % confThresh = 0.01;
    confThresh = -Inf;
    fgThresh = 0.5;
    bgThreshLo = 0;
    bgThreshHi = 0.5;
    nSubclass = 10 * ones(1, 20); % cluster size: 1xC
    bboxMean = zeros(4, 20); % mean: 4xC
    bboxStd = [0.1 * ones(2, 20); 0.2 * ones(2, 20)]; % std: 4xC
    bboxMean2 = zeros(4, 1);
    bboxStd2 = [0.1; 0.1; 0.2; 0.2];
    subclassPos = 16;
    subclassNeg = 48;
    keep_neg_n = +Inf; % train
    top_n = 100; % test
    singleRegress = true; % single regress for each shape category
    testProbBg = false; % include bg probability in test

    % debug
    DEBUG = false;
  end

  methods
    % inputs:
    % inputs{1} probcls: 1 x 1 x (C+1) x P
    % inputs{2} predbbox: 1 x 1 x 4(C+1) x P
    % inputs{3} rois: 5 x P selective search proposals [gpuArray]
    % inputs{4} gtboxes: [left, top, right, bottom, class, shape(s)]
    %               multi-class: 6 x Pgt
    %               multi-label: [5 + (max(Csub) + 1)] x Pgt (filled with nan if variable label length) (+1 stands for the bkg label)
    % inputs{5} imsize: [h, w] 1 x 2 (each batch has the same image size)

    % params:
    % none

    % outputs:
    % outputs{1-C} rois_c: 5 x Pc [gpuArray]
    % outputs{(C+1)-2C} [TRAIN] label_c: 1 x Pc (multi-class)
    %                                    1 x 1 x (Csub + 1) x Pc (multi-label)
    %                   [TEST] probs_c: 1 x min(TOP_N, P)
    % outputs{(2C+1)-3C} targets_c: 1x1x 4(Csub+1) x Pc [gpuArray]
    % outputs{(3C+1)-4C} instance_weights_c: 1x1x 4(Csub+1) x Pc [gpuArray]

    function outputs = forward(obj, inputs, params)
        cls_probs = gather(squeeze(inputs{1}));
        box_deltas = gather(squeeze(inputs{2})); % 4(C+1) x P

        % normalize
        box_deltas = bsxfun(@times, box_deltas, ...
                        [obj.bboxStd(:); zeros(4, 1, 'single')]);
        box_deltas = bsxfun(@plus, box_deltas, ...
                            [obj.bboxMean(:); zeros(4, 1, 'single')]);

        rois = gather(inputs{3}); % [5, P]
        gtboxes = inputs{4};
        imsize = inputs{5};

        testMode = all(isnan(gtboxes(:)));
        
        % during test time, when there's no GTBOXES, we mark GTBOXES as nan (so that the layer can still be evaluated
        % in this case we want to pass all boxes exceeding CONFTHRESH 
        if testMode 
            gtboxes = nan(6, 0);
        end

        C = size(cls_probs, 1) - 1; % 20
        B = max(rois(1, :)); % batchsize
        assert(B == 1);

        gt_classes = gtboxes(5, :);
        gt_subclasses = gtboxes(6, :);
        
        outputs = cell(1, 4*C);

        for c = 1 : C
            bglabel = obj.nSubclass(c) + 1;
            % pick proposals for class C
            probs_c = cls_probs(c, :);
            roi_select = (probs_c > obj.confThresh);
            
            if ~any(roi_select), continue; end % no proposals for a class at all
            rois_c = bbox_transform_inv(rois(2:end, roi_select)', box_deltas(4*(c-1)+ 1 : 4*c, roi_select)')'; % output from BBOX_TRANSFORM_INV: Px4
            % clip
            rois_c = bbox_clip(round(rois_c'), imsize)';

            probs_c = probs_c(roi_select);
            if obj.testProbBg
                probs_c = cls_probs([c, end], roi_select); % include bkg
            end
            S = size(rois_c, 2);

            if testMode
                if S > obj.top_n
                    [~, order] = sort(probs_c(1, :), 'descend');
                    keep = order(1:obj.top_n);
                else
                    keep = (1:S);
                end

                rois_c = [ones(1, numel(keep)); rois_c(:, keep)];
                outputs{c} = gpuArray(rois_c);
                outputs{C + c} = probs_c(:, keep);
                continue;
            end
            gt_select = (gt_classes == c);
            % if ~any(gt_select), continue; end % no gt, then no pos and no neg as ovlp = 0

            gtboxes_c = gtboxes(1:4, gt_select);
            gt_subclasses_c = gt_subclasses(:, gt_select);

            targets_c = zeros(S, 4);

            ovlp = bbox_overlap(rois_c', gtboxes_c');
            [ious_c, gt_assignments] = max(ovlp, [], 2);
            
            labels_c = gt_subclasses_c(:, gt_assignments);


            if ~any(gt_select) % no iou computed
                pos = [];
                neg = 1 : S;
            else
                % targets for fg only
                pos = find(ious_c >= obj.fgThresh);
                % if numel(pos) == 0, continue; end % if no pos, then no neg
                gtboxes_c_correspond = gtboxes_c(:, gt_assignments(pos));
                targets_c(pos, :) = bbox_transform(rois_c(:, pos)', gtboxes_c_correspond'); % output from BBOX_TRANFORM: Px4
                neg = find(ious_c >= obj.bgThreshLo & ious_c < obj.bgThreshHi);
            end

            npos = numel(pos);
            nneg = numel(neg);

            subclassPos = min(npos, obj.subclassPos);
            subclassAll = obj.subclassPos + obj.subclassNeg;
            subclassNeg = min(nneg, subclassAll - subclassPos);

            % loss sampling (pos)
            if npos > subclassPos
                pos = pos(randsample(npos, subclassPos)); % last stage: remove excessive pos
            end

            if nneg > subclassNeg
                probs_c_neg = probs_c(neg);
                if nneg > obj.keep_neg_n % pick from top KEEP_NEG_N
                    [~, order] = sort(probs_c_neg, 'descend');
                    neg = neg(order(1 : obj.keep_neg_n));
                end
                % loss sampling (neg)
                % no need for pathway sampling, because subclass is the last (3rd) stage
                neg = neg(randsample(numel(neg), subclassNeg));
            end
                
            if npos > 0 && obj.DEBUG
                fprintf('[proposal2] cat #%d - loss sampling -- pos: %d/%d | neg: %d/%d\n', c, numel(pos), npos, ...
                    numel(neg), nneg);
                % keyboard
            end

            labels_c(neg) = bglabel;

            labels_c = labels_c(:, [pos; neg]);
            rois_c = rois_c(:, [pos; neg]);
            rois_c = [ones(1, numel(labels_c)); rois_c];
            % ious_c = ious_c([pos; neg]);
            targets_c = targets_c([pos; neg], :);

            targets_c = bsxfun(@minus, targets_c, obj.bboxMean2');
            targets_c = bsxfun(@rdivide, targets_c, obj.bboxStd2');

            outputs{c} = gpuArray(rois_c);
            outputs{C + c} = labels_c;
            % if npos == 0, keyboard; end
                
            R = size(rois_c, 2); % total number of ROIs
            if obj.singleRegress
                % multi-class
                targets = zeros(1, 1, 4, R, 'single');
                instance_weights = zeros(1, 1, 4, R, 'single');
                for r = 1 : R
                    lb = labels_c(r);
                    if (lb < bglabel)
                        targets(:, :, :, r) = targets_c(r, :);
                        instance_weights(:, :, :, r) = 1;
                    end
                end
            else
                targets = zeros(1, 1, 4 * (obj.nSubclass(c) + 1), R, 'single');
                instance_weights = zeros(1, 1, 4 * (obj.nSubclass(c) + 1), R, 'single');

                for r = 1 : R
                    lb = labels_c(r);
                    assert(lb > 0);
                    if (lb < bglabel)
                        targets(1, 1, 4 * (lb - 1) + 1 : 4 * lb, r) = targets_c(r, :);
                        instance_weights(1, 1, 4 * (lb - 1) + 1 : 4 * lb, r) = 1;
                    end
                end
            end

            outputs{2 * C + c} = gpuArray(targets);
            outputs{3 * C + c} = gpuArray(instance_weights);
            if obj.DEBUG
                fprintf('[proposal2]\n'); 
                keyboard
            end
        end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      % do not back-prop
      for i = 1 : 4
        derInputs{i} = [];
      end

      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
        error('Not impelmented: dynamic output size.\n');
    end

    function obj = Proposal2(varargin)
      obj.load(varargin) ;
    end
  end
end
