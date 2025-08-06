module Torch
  module Distributions
    class Normal < ExponentialFamily
      def initialize(loc, scale, validate_args: nil)
        @loc, @scale = Utils.broadcast_all(loc, scale)
        if loc.is_a?(Numeric) && scale.is_a?(Numeric)
          batch_shape = []
        else
          batch_shape = @loc.size
        end
        super(batch_shape:, validate_args:)
      end

      def sample(sample_shape: [])
        shape = _extended_shape(sample_shape:)
        Torch.no_grad do
          Torch.normal(@loc.expand(shape), @scale.expand(shape))
        end
      end
    end
  end
end
