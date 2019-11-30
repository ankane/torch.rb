module Torch
  module NN
    class Functional
      class << self
        def relu(input)
          Torch.relu(input)
        end

        def conv2d(input, weight, bias, stride: 1, padding: 0, dilation: 1, groups: 1)
          # TODO pair stride and padding when needed
          Torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
        end

        def max_pool2d(input, kernel_size)
          kernel_size = [kernel_size, kernel_size] if kernel_size.is_a?(Integer)
          Torch.max_pool2d(input, kernel_size)
        end

        def avg_pool2d(input, kernel_size)
          kernel_size = [kernel_size, kernel_size] if kernel_size.is_a?(Integer)
          Torch.avg_pool2d(input, kernel_size)
        end

        def linear(input, weight, bias)
          Torch.linear(input, weight, bias)
        end

        def mse_loss(input, target, reduction: "mean")
          Torch.mse_loss(input, target, reduction)
        end

        def cross_entropy(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          nll_loss(log_softmax(input, 1), target, weight: weight, ignore_index: ignore_index, reduction: reduction)
        end

        def nll_loss(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          raise NotImplementedYet if weight
          Torch.nll_loss(input, target, reduction, ignore_index)
        end

        def l1_loss(input, target, reduction: "mean")
          Torch.l1_loss(input, target, reduction)
        end

        def log_softmax(input, dim)
          input.log_softmax(dim)
        end

        def dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch._dropout!(input, p, training)
          else
            Torch._dropout(input, p, training)
          end
        end

        def dropout2d(input, p: 0.5, training: true, inplace: false)
          raise ArgumentError, "dropout probability has to be between 0 and 1, but got #{p}" if p < 0 || p > 1

          if inplace
            Torch._feature_dropout!(input, p, training)
          else
            Torch._feature_dropout(input, p, training)
          end
        end

        def dropout3d(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch._feature_dropout!(input, p, training)
          else
            Torch._feature_dropout(input, p, training)
          end
        end

        def alpha_dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch._alpha_dropout!(input, p, training)
          else
            Torch._alpha_dropout(input, p, training)
          end
        end

        def feature_alpha_dropout(input, p: 0.5, training: true, inplace: false)
          if inplace
            Torch._feature_alpha_dropout!(input, p, training)
          else
            Torch._feature_alpha_dropout(input, p, training)
          end
        end

        def embedding(input, weight, padding_idx: nil, max_norm: nil, norm_type: 2.0, scale_grad_by_freq: false, sparse: false)
          # TODO handle max_norm and norm_type
          raise NotImplementedYet unless max_norm.nil? && norm_type == 2.0

          padding_idx ||= -1
          Torch._embedding(input, weight, padding_idx, scale_grad_by_freq, sparse)
        end
      end
    end

    # shortcut
    F = Functional
  end
end
