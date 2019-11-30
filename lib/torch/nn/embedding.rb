# ported from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py
module Torch
  module NN
    class Embedding < Module
      def initialize(num_embeddings, embedding_dim, padding_idx: nil, max_norm: nil,
        norm_type: 2.0, scale_grad_by_freq: false, sparse: false, _weight: nil)

        super()
        @num_embeddings = num_embeddings
        @embedding_dim = embedding_dim

        if padding_idx
          if padding_idx > 0
            raise ArgumentError, "Padding_idx must be within num_embeddings" unless padding_idx < @num_embeddings
          elsif padding_idx < 0
            raise ArgumentError, "Padding_idx must be within num_embeddings" unless padding_idx >= -@num_embeddings
            padding_idx = @num_embeddings + padding_idx
          end
        end
        @padding_idx = padding_idx
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
        @sparse = sparse
      end

      def reset_parameters
        Init.normal!(@weight)
        if @padding_idx
          Torch.no_grad do
            @weight[@padding_idx].fill!(0)
          end
        end
      end

      def forward(input)
        F.embedding(input, @weight, padding_idx: @padding_idx, max_norm: @max_norm, norm_type: @norm_type, scale_grad_by_freq: @scale_grad_by_freq, sparse: @sparse)
      end
    end
  end
end
