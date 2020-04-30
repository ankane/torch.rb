module Torch
  module NN
    class BatchNorm < Module
      def initialize(num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true)
        super()
        @num_features = num_features
        @eps = eps
        @momentum = momentum
        @affine = affine
        @track_running_stats = track_running_stats
        if @affine
          @weight = Parameter.new(Torch::Tensor.new(num_features))
          @bias = Parameter.new(Torch::Tensor.new(num_features))
        else
          register_parameter("weight", nil)
          register_parameter("bias", nil)
        end
        if track_running_stats
          register_buffer("running_mean", Torch.zeros(num_features))
          register_buffer("running_var", Torch.ones(num_features))
          register_buffer("num_batches_tracked", Torch.tensor(0, dtype: :long))
        else
          register_parameter("running_mean", nil)
          register_parameter("running_var", nil)
          register_parameter("num_batches_tracked", nil)
        end
        reset_parameters
      end

      def reset_running_stats
        if @track_running_stats
          @running_mean.zero!
          @running_var.fill!(1)
          @num_batches_tracked.zero!
        end
      end

      def reset_parameters
        reset_running_stats
        if @affine
          Init.ones!(@weight)
          Init.zeros!(@bias)
        end
      end

      def forward(input)
        _check_input_dim(input)

        if @momentum.nil?
          exponential_average_factor = 0.0
        else
          exponential_average_factor = @momentum
        end

        if @training and @track_running_stats
          if @num_batches_tracked.nil?
            @num_batches_tracked += 1
            if @momentum.nil?
              exponential_average_factor = 1.0 / @num_batches_tracked.to_f
            else
              exponential_average_factor = @momentum
            end
          end
        end

        F.batch_norm(
          input, @running_mean, @running_var,
          weight: @weight, bias: @bias,
          training: @training || !@track_running_stats,
          momentum: exponential_average_factor, eps: @eps
        )
      end

      def extra_inspect
        s = "%{num_features}, eps: %{eps}, momentum: %{momentum}, affine: %{affine}, track_running_stats: %{track_running_stats}"
        format(s, **dict)
      end
    end
  end
end
