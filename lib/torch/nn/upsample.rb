module Torch
  module NN
    class Upsample < Module
      def initialize(size: nil, scale_factor: nil, mode: "nearest", align_corners: nil)
        super()
        @size = size
        if scale_factor.is_a?(Array)
          @scale_factor = scale_factor.map(&:to_f)
        else
          @scale_factor = scale_factor ? scale_factor.to_f : nil
        end
        @mode = mode
        @align_corners = align_corners
      end

      def forward(input)
        F.interpolate(input, size: @size, scale_factor: @scale_factor, mode: @mode, align_corners: @align_corners)
      end

      def extra_inspect
        if !@scale_factor.nil?
          info = "scale_factor: #{@scale_factor.inspect}"
        else
          info = "size: #{@size.inspect}"
        end
        info += ", mode: #{@mode.inspect}"
        info
      end
    end
  end
end
