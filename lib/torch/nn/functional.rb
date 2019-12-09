module Torch
  module NN
    class Functional
      class << self
        def relu(input, inplace: false)
          if inplace
            input.relu!
          else
            input.relu
          end
        end

        # convolution layers

        def conv1d(*args, **options)
          Torch.conv1d(*args, **options)
        end

        def conv2d(*args, **options)
          Torch.conv2d(*args, **options)
        end

        def conv3d(*args, **options)
          Torch.conv3d(*args, **options)
        end

        def unfold(input, kernel_size, dilation: 1, padding: 0, stride: 1)
          if input.dim == 4
            NN.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
          else
            raise Error, "Input Error: Only 4D input Tensors are supported (got #{input.dim}D)"
          end
        end

        def fold(input, output_size, kernel_size, dilation: 1, padding: 0, stride: 1)
          if input.dim == 3
            NN.col2im(input, _pair(output_size), _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
          else
            raise Error, "Input Error: Only 3D input Tensors are supported (got #{input.dim}D)"
          end
        end

        # pooling layers

        def max_pool1d(*args, **options)
          Torch.max_pool1d(*args, **options)
        end

        def max_pool2d(*args, **options)
          Torch.max_pool2d(*args, **options)
        end

        def max_pool3d(*args, **options)
          Torch.max_pool3d(*args, **options)
        end

        def avg_pool1d(*args, **options)
          NN.avg_pool1d(*args, **options)
        end

        def avg_pool2d(*args, **options)
          NN.avg_pool2d(*args, **options)
        end

        def avg_pool3d(*args, **options)
          NN.avg_pool3d(*args, **options)
        end

        # other layers

        def prelu(input, weight)
          Torch.prelu(input, weight)
        end

        def leaky_relu(input, negative_slope = 0.01)
          NN.leaky_relu(input, negative_slope)
        end

        # linear layers

        def bilinear(input1, input2, weight, bias)
          Torch.bilinear(input1, input2, weight, bias)
        end

        def linear(input, weight, bias)
          NN.linear(input, weight, bias)
        end

        # sparse layers

        def embedding(input, weight, padding_idx: nil, max_norm: nil, norm_type: 2.0, scale_grad_by_freq: false, sparse: false)
          # TODO handle max_norm and norm_type
          raise NotImplementedYet unless max_norm.nil? && norm_type == 2.0

          padding_idx ||= -1
          # weight and indices are swapped from Python interface
          Torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
        end

        def embedding_bag(input, weight, offsets: nil, max_norm: nil, norm_type: 2, scale_grad_by_freq: false, mode: "mean", sparse: false, per_sample_weights: nil)
          # need to handle nils
          raise NotImplementedYet

          # TODO handle max_norm and norm_type
          raise NotImplementedYet unless max_norm.nil? && norm_type == 2.0

          Torch.embedding_bag(input, weight, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights)
        end

        # distance functions

        def cosine_similarity(x1, x2, dim: 1, eps: 1e-8)
          Torch.cosine_similarity(x1, x2, dim, eps)
        end

        def pairwise_distance(x1, x2, p: 2.0, eps: 1e-6, keepdim: false)
          Torch.pairwise_distance(x1, x2, p, eps, keepdim)
        end

        # loss functions

        def binary_cross_entropy(input, target, weight: nil, reduction: "mean")
          NN.binary_cross_entropy(input, target, weight, reduction)
        end

        def binary_cross_entropy_with_logits(input, target, weight: nil, reduction: "mean", pos_weight: nil)
          Torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)
        end

        def cosine_embedding_loss(input1, input2, target, margin: 0, reduction: "mean")
          raise NotImplementedYet
        end

        def cross_entropy(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          nll_loss(log_softmax(input, 1), target, weight: weight, ignore_index: ignore_index, reduction: reduction)
        end

        def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank: 0, reduction: "mean", zero_infinity: false)
          # call to_a on input_lengths and target_lengths for C++
          Torch.ctc_loss(log_probs, targets, input_lengths.to_a, target_lengths.to_a, blank, reduction, zero_infinity)
        end

        def hinge_embedding_loss(input, target, margin: 1.0, reduction: "mean")
          Torch.hinge_embedding_loss(input, target, margin, reduction)
        end

        def kl_div(input, target, reduction: "mean")
          Torch.kl_div(input, target, reduction)
        end

        def l1_loss(input, target, reduction: "mean")
          NN.l1_loss(input, target, reduction)
        end

        def margin_ranking_loss(input1, input2, target, margin: 0, reduction: "mean")
          raise NotImplementedYet
        end

        def mse_loss(input, target, reduction: "mean")
          NN.mse_loss(input, target, reduction)
        end

        def multilabel_margin_loss(input, target, reduction: "mean")
          NN.multilabel_margin_loss(input, target, reduction)
        end

        def multilabel_soft_margin_loss(input, target, weight: nil)
          raise NotImplementedYet
        end

        def multi_margin_loss(input, target, p: 1, margin: 1.0, weight: nil, reduction: "mean")
          NN.multi_margin_loss(input, target, p, margin, weight, reduction)
        end

        def nll_loss(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          NN.nll_loss(input, target, weight, reduction, ignore_index)
        end

        def poisson_nll_loss(input, target, log_input: true, full: false, eps: 1e-8, reduction: "mean")
          Torch.poisson_nll_loss(input, target, log_input, full, eps, reduction)
        end

        def soft_margin_loss(input, target, reduction: "mean")
          NN.soft_margin_loss(input, target, reduction)
        end

        def smooth_l1_loss(input, target, reduction: "mean")
          NN.smooth_l1_loss(input, target, reduction)
        end

        def triplet_margin_loss(anchor, positive, negative, margin: 1.0, p: 2, eps: 1e-06, swap: false, reduction: "mean")
          Torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction)
        end

        # end loss

        def softmax(input, dim: nil)
          dim ||= softmax_dim(input.dim)
          input.softmax(dim)
        end

        def softmin(input, dim: nil)
          dim ||= softmax_dim(input.dim)
          (-input).softmax(dim)
        end

        def softplus(input, beta: 1, threshold: 20)
          NN.softplus(input, beta, threshold)
        end

        # TODO make dim keyword argument and update examples
        def log_softmax(input, dim = nil)
          dim ||= softmax_dim(input.dim)
          input.log_softmax(dim)
        end

        def dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch.dropout!(input, p, training)
          else
            Torch.dropout(input, p, training)
          end
        end

        def dropout2d(input, p: 0.5, training: true, inplace: false)
          raise ArgumentError, "dropout probability has to be between 0 and 1, but got #{p}" if p < 0 || p > 1

          if inplace
            Torch.feature_dropout!(input, p, training)
          else
            Torch.feature_dropout(input, p, training)
          end
        end

        def dropout3d(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch.feature_dropout!(input, p, training)
          else
            Torch.feature_dropout(input, p, training)
          end
        end

        def alpha_dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch.alpha_dropout!(input, p, training)
          else
            Torch.alpha_dropout(input, p, training)
          end
        end

        def feature_alpha_dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch.feature_alpha_dropout!(input, p, training)
          else
            Torch.feature_alpha_dropout(input, p, training)
          end
        end

        private

        def softmax_dim(ndim)
          ndim == 0 || ndim == 1 || ndim == 3 ? 0 : 1
        end

        def _pair(value)
          value.is_a?(Array) ? value : [value] * 2
        end
      end
    end

    # shortcut
    F = Functional
  end
end
