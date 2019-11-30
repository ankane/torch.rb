# ported from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py
module Torch
  module NN
    class EmbeddingBag < Module
      def initialize(num_embeddings, embedding_dim, max_norm: nil, norm_type: 2.0,
        scale_grad_by_freq: false, mode: "mean", sparse: false, _weight: nil)

        super()
        @num_embeddings = num_embeddings
        @embedding_dim = embedding_dim
        @max_norm = max_norm
        @norm_type = norm_type
        @scale_grad_by_freq = scale_grad_by_freq
        if _weight.nil?
          @weight = Parameter.new(Tensor.new(num_embeddings, embedding_dim))
          reset_parameters
        else
          raise ArgumentError, "Shape of weight does not match num_embeddings and embedding_dim" unless _weight.shape == [num_embeddings, embedding_dim]
          @weight = Parameter.new(_weight)
        end
        @mode = mode
        @sparse = sparse
      end

      def reset_parameters
        Init.normal!(@weight)
      end

      def forward(input, offsets: nil, per_sample_weights: nil)
        F.embedding_bag(input, @weight, offsets: offsets, max_norm: @max_norm, norm_type: @norm_type, scale_grad_by_freq: @scale_grad_by_freq, mode: @mode, sparse: @sparse, per_sample_weights: per_sample_weights)
      end
    end
  end
end
