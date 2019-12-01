module Torch
  module NN
    module Init
      class << self
        def calculate_gain(nonlinearity, param: 0.01)
          _calculate_gain(nonlinearity, param)
        end

        def uniform!(tensor, a: 0.0, b: 1.0)
          _uniform!(tensor, a, b)
        end

        def normal!(tensor, mean: 0.0, std: 1.0)
          _normal!(tensor, mean, std)
        end

        def constant!(tensor, val)
          _constant!(tensor, val)
        end

        def ones!(tensor)
          _ones!(tensor)
        end

        def zeros!(tensor)
          _zeros!(tensor)
        end

        def eye!(tensor)
          _eye!(tensor)
        end

        def dirac!(tensor)
          _dirac!(tensor)
        end

        def xavier_uniform!(tensor, gain: 1.0)
          _xavier_uniform!(tensor, gain)
        end

        def xavier_normal!(tensor, gain: 1.0)
          _xavier_normal!(tensor, gain)
        end

        def kaiming_uniform!(tensor, a: 0, mode: "fan_in", nonlinearity: "leaky_relu")
          _kaiming_uniform!(tensor, a, mode, nonlinearity)
        end

        def kaiming_normal!(tensor, a: 0, mode: "fan_in", nonlinearity: "leaky_relu")
          _kaiming_normal!(tensor, a, mode, nonlinearity)
        end

        def orthogonal!(tensor, gain: 1)
          _orthogonal!(tensor, gain)
        end

        def sparse!(tensor, sparsity, std: 0.01)
          _sparse!(tensor, sparsity, std)
        end

        # TODO move to C++ when released
        def _calculate_fan_in_and_fan_out(tensor)
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
