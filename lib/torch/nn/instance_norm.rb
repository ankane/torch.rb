module Torch
  module NN
    class InstanceNorm < BatchNorm
      def initialize(num_features, eps: 1e-5, momentum: 0.1, affine: false, track_running_stats: false)
        super(num_features, eps: eps, momentum: momentum, affine: affine, track_running_stats: track_running_stats)
      end

      def forward(input)
        _check_input_dim(input)

        F.instance_norm(
          input, running_mean: @running_mean, running_var: @running_var,
          weight: @weight, bias: @bias,
          use_input_stats: @training || !@track_running_stats,
          momentum: @momentum, eps: @eps
        )
      end
    end
  end
end
