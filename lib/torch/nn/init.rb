module Torch
  module NN
    module Init
      class << self
        def calculate_fan_in_and_fan_out(tensor)
          dimensions = tensor.dim
          if dimensions < 2
            raise Error, "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
          end

          if dimensions == 2
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
          else
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim > 2
              receptive_field_size = tensor[0][0].numel
            end
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
          end

          [fan_in, fan_out]
        end
      end
    end
  end
end
