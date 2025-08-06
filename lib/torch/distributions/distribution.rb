module Torch
  module Distributions
    class Distribution
      def initialize(batch_shape: [], event_shape: [], validate_args: nil)
        @batch_shape = batch_shape
        @event_shape = event_shape
        if !validate_args.nil?
          @validate_args = validate_args
        end
        if @validate_args
          raise NotImplementedYet
        end
        super()
      end

      private

      def _extended_shape(sample_shape: [])
        if !sample_shape.is_a?(Array)
          sample_shape = sample_shape.to_a
        end
        sample_shape + @batch_shape + @event_shape
      end
    end
  end
end
