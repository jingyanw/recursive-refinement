classdef Split2 < dagnn.ElementWise
%SPLIT DagNN splitting layer.
%    This layer splits inputs{1} according to inputs{2} (SPLIT)
  properties
    dim = 4
  end

  methods
    function outputs = forward(obj, inputs, params)
      split = inputs{2};
      assert(size(inputs{1}, obj.dim) == sum(split));

      s = num2cell(size(inputs{1}));
      s{obj.dim} = split;
      if length(size(inputs{1})) == 3 % corner case: singleton
        idx = find(split);
        assert(numel(idx) == 1);
        outputs = cell(1, numel(split));
        outputs{idx} = inputs{1};
      else
        outputs = mat2cell(inputs{1}, s{:});
        outputs = squeeze(outputs);
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = cat(obj.dim, derOutputs{:});
      derInputs{2} = [];
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      error('getOutputSizes not implemented for SPLIT2 layer\n');
    end

    function rfs = getReceptiveFields(obj)
        error('getReceptiveFields not implemented for SPLIT2 layer\n');
    end

    function obj = Split2(varargin)
      obj.load(varargin) ;
    end
  end
end
