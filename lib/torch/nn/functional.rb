module Torch
  module NN
    class Functional
      class << self
        def relu(input)
          Torch.relu(input)
        end

        def conv2d(input, weight, bias)
          Torch.conv2d(input, weight, bias)
        end

        def max_pool2d(input, kernel_size)
          kernel_size = [kernel_size, kernel_size] if kernel_size.is_a?(Integer)
          Torch.max_pool2d(input, kernel_size)
        end

        def linear(input, weight, bias)
          Torch.linear(input, weight, bias)
        end

        def mse_loss(input, target, reduction: "mean")
          Torch.mse_loss(input, target, reduction)
        end

        def cross_entropy(input, target)
          nll_loss(log_softmax(input, 1), target)
        end

        def nll_loss(input, target)
          # TODO fix for non-1d
          Torch.nll_loss(input, target)
        end

        def log_softmax(input, dim)
          input.log_softmax(dim)
        end
      end
    end

    # shortcut
    F = Functional
  end
end
