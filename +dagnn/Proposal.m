classdef Proposal < dagnn.Layer
  
  properties
    nmsThresh = 0.7;
    pre_nms_top_n = [12000, 6000];
    post_nms_top_n = [2000, 300];
    keep_pos_n = +Inf;
    keep_neg_n = 500;
    % min_size = 16; % remove proposals smaller than 16x16
    nClass = 20;
    bboxMean = zeros(1, 4);
    bboxStd = ones(1, 4);
    bboxMean2 = zeros(1, 4);
    bboxStd2 = [0.1, 0.1, 0.2, 0.2];
    anchors = generate_anchors();
    feat_stride = 16;

    classPos = 128;
    classNeg = 128;
    fgThresh = 0.5;
    bgThreshHi = 0.5;
    bgThreshLo = 0;

    % debug
    debug = false;
  end

  methods
    % Inputs:
    % inputs{1} rpn_prob: [H, W, A]
    % inputs{2} rpn_reg: [H, W, 4A]
    % inputs{3} gtboxes: [left, top, right, bottom, class, shape(s)]
    %   multi-class: 6 x G
    % inputs{4} imsize: [h, w] 1 x 2

    % params:
    % none

    % Outputs:
    % outputs{1} rois: 5 x P [gpuArray]
    % outputs{2} [TRAIN] label: 1 x P (multi-class)
    %            [TEST] probs: 1 x P (probabilities)
    % outputs{3} [TRAIN only] targets: [1, 1, 4, P]
    % outputs{4} [TRAIN only] instance_weights: [1, 1, 4, P]

    function outputs = forward(obj, inputs, params)
        useGPU = isa(inputs{1}, 'gpuArray');
        rpn_prob = gather(inputs{1});
        rpn_reg = gather(inputs{2});
        gtboxes = inputs{3};
        imsize = inputs{4};
        [H, W, A] = size(rpn_prob);
        G = size(gtboxes, 2);

        % decide mode
        if ~isnan(gtboxes) % train + val
            mode = 1;
        else % test
            mode = 2;
        end
        pre_nms_top_n = obj.pre_nms_top_n(mode);
        post_nms_top_n = obj.post_nms_top_n(mode);

        % enumerate anchor boxes
        shift_x = (0 : W - 1) * obj.feat_stride;
        shift_y = (0 : H - 1) * obj.feat_stride;
        [shift_x, shift_y] = meshgrid(shift_x, shift_y);
        shifts = cat(4, shift_x, shift_y, shift_x, shift_y);
        anchors = reshape(obj.anchors, [1, 1, A, 4]);

        all_anchors = bsxfun(@plus, anchors, shifts); % [H, W, A, 4]

        rpn_reg = target_transform_inv(rpn_reg);
        reg = reshape(rpn_reg, [], 4);

        % unnormalize
        reg = bsxfun(@times, reg, obj.bboxStd);
        reg = bsxfun(@plus, reg, obj.bboxMean);

        rpn_prob = rpn_prob(:);

        % top PRE_NMS_TOP_N
        boxes = reshape(all_anchors, [], 4);

        anchor_idxs = repelem(1:A, H * W); % keep track of the anchor type for each box
        if pre_nms_top_n < numel(rpn_prob)
            [~, order] = sort(rpn_prob, 'descend');
            select = order(1:pre_nms_top_n);
            boxes = boxes(select, :);
            reg = reg(select, :);
            rpn_prob = rpn_prob(select);
            anchor_idxs = anchor_idxs(select);
        end

        % apply regression
        proposals = bbox_transform_inv(boxes, reg);
        proposals = bbox_clip(proposals, imsize);

        % NMS
        pick = nms([proposals, rpn_prob], obj.nmsThresh, useGPU);
        proposals = proposals(pick, :);
        rpn_prob = rpn_prob(pick);
        anchor_idxs = anchor_idxs(pick);

        % top POST_NMS_TOP_N
        if post_nms_top_n < numel(rpn_prob)
            proposals = proposals(1:post_nms_top_n, :);
            rpn_prob = rpn_prob(1:post_nms_top_n, :);
            anchor_idxs = anchor_idxs(1:post_nms_top_n);
        end
        
        switch mode
            case 2 % TEST
                P = size(proposals, 1);
                rois = [ones(1, P); proposals'];

                outputs{1} = gpuArray(rois);
                outputs{2} = rpn_prob;
            case 1 % TRAIN
                ovlp = bbox_overlap(proposals, gtboxes(1:4, :)'); % [#proposals, G]
                [max_iou, gt_assignments] = max(ovlp, [], 2);
                labels_all = gtboxes(5, gt_assignments);

                pos = find(max_iou >= obj.fgThresh);
                neg = find((max_iou < obj.bgThreshHi) & (max_iou >= obj.bgThreshLo));
                % passway sampling
                if numel(pos) > obj.keep_pos_n
                    pos = datasample(pos, obj.keep_pos_n, 'Replace', false);
                end
                if numel(neg) > obj.keep_neg_n
                    neg = datasample(neg, obj.keep_neg_n, 'Replace', false);
                end

                P = numel(pos) + numel(neg);
                rois = [ones(1, P); proposals(pos, :)', proposals(neg, :)'];
                anchor_idxs = [anchor_idxs(pos), anchor_idxs(neg)]; % work here

                % loss sampling
                if numel(pos) > obj.classPos
                    pos_loss_idx = randsample(numel(pos), obj.classPos); % w/o replacement
                else
                    pos_loss_idx = (1 : numel(pos));
                end

                classNeg = max(obj.classPos + obj.classNeg - numel(pos), obj.classNeg);

                if numel(neg) > classNeg
                    neg_loss_idx = randsample(numel(neg), classNeg); % w/o replacement
                else
                    neg_loss_idx = (1 : numel(neg));
                end
                boxes_pos = proposals(pos(pos_loss_idx), :);
                boxes_gt = gtboxes(1:4, gt_assignments(pos(pos_loss_idx)))';
                t = bbox_transform(boxes_pos, boxes_gt);

                % normalize
                t = bsxfun(@minus, t, obj.bboxMean2);
                t = bsxfun(@rdivide, t, obj.bboxStd2);

                if P == 0
                    labels = zeros(0, 0, 'single');
                    targets = zeros(0, 0, 'single');
                    instance_weights = zeros(0, 0, 'single');
                else
                    labels = zeros(1, P);
                    labels(pos_loss_idx) = labels_all(pos(pos_loss_idx));
                    labels(numel(pos) + neg_loss_idx) = obj.nClass + 1; % bkg

                    targets = gpuArray(zeros(1, 1, 4 * (obj.nClass + 1), P, 'single'));
                    instance_weights = gpuArray(zeros(1, 1, 4 * (obj.nClass + 1), P, 'single'));
                    for i = 1 : numel(pos_loss_idx)
                        lb = labels_all(pos(pos_loss_idx(i)));
                        targets(1, 1, 4 * (lb-1) + 1 : 4 * lb, pos_loss_idx(i)) = t(i, :);
                        instance_weights(1, 1, 4 * (lb-1) + 1 : 4*lb, pos_loss_idx(i)) = 1;
                    end
                end
              
                outputs{1} = gpuArray(rois);
                outputs{2} = labels;
                outputs{3} = targets;
                outputs{4} = instance_weights;
                
                if obj.debug
                    fprintf('[proposal] pos: %d | neg: %d\n', ...
                        sum((labels < obj.nClass + 1) & (labels > 0)), ...
                        sum(labels == obj.nClass + 1) ...
                    );
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

    function obj = Proposal(varargin)
      obj.load(varargin) ;
    end
  end
end
