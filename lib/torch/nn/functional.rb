module Torch
  module NN
    class Functional
      class << self
        include Utils

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
          return_indices = args.pop if args.size == 7
          if return_indices
            Torch.max_pool1d_with_indices(*args, **options)
          else
            Torch.max_pool1d(*args, **options)
          end
        end

        def max_pool2d(*args, **options)
          return_indices = args.pop if args.size == 7
          if return_indices
            NN.max_pool2d_with_indices(*args, **options)
          else
            Torch.max_pool2d(*args, **options)
          end
        end

        def max_pool3d(*args, **options)
          return_indices = args.pop if args.size == 7
          if return_indices
            NN.max_pool3d_with_indices(*args, **options)
          else
            Torch.max_pool3d(*args, **options)
          end
        end

        def max_unpool1d(input, indices, kernel_size, stride: nil, padding: 0, output_size: nil)
          raise NotImplementedYet
          kernel_size = _single(kernel_size)
          if !stride.nil?
            _stride = _single(stride)
          else
            _stride = kernel_size
          end
          padding = _single(padding)
          output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
          output_size = output_size + [1]
          NN.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3), output_size).squeeze(3)
        end

        def max_unpool2d(*args, **options)
          raise NotImplementedYet
          NN.max_unpool2d(*args, **options)
        end

        def max_unpool3d(*args, **options)
          raise NotImplementedYet
          NN.max_unpool3d(*args, **options)
        end

        def avg_pool1d(*args, **options)
          Torch.avg_pool1d(*args, **options)
        end

        def avg_pool2d(*args, **options)
          NN.avg_pool2d(*args, **options)
        end

        def avg_pool3d(*args, **options)
          NN.avg_pool3d(*args, **options)
        end

        def adaptive_max_pool1d(*args, **options)
          Torch.adaptive_max_pool1d(*args, **options)
        end

        def adaptive_max_pool2d(input, output_size)
          output_size = list_with_default(output_size, input.size)
          NN.adaptive_max_pool2d(input, output_size)
        end

        def adaptive_max_pool3d(input, output_size)
          output_size = list_with_default(output_size, input.size)
          NN.adaptive_max_pool3d(input, output_size)
        end

        def adaptive_avg_pool1d(*args, **options)
          Torch.adaptive_avg_pool1d(*args, **options)
        end

        def adaptive_avg_pool2d(input, output_size)
          output_size = list_with_default(output_size, input.size)
          NN.adaptive_avg_pool2d(input, output_size)
        end

        def adaptive_avg_pool3d(input, output_size)
          output_size = list_with_default(output_size, input.size)
          NN.adaptive_avg_pool3d(input, output_size)
        end

        # padding layers

        def pad(input, pad, mode: "constant", value: 0)
          raise ArgumentError, "Padding length must be divisible by 2" unless pad.size % 2 == 0
          raise ArgumentError, "Padding length too large" unless pad.size / 2 <= input.dim

          if mode == "constant"
            Torch.constant_pad_nd(input, pad, value)
          else
            raise ArgumentError, "Padding mode doesn't take in value argument" unless value == 0

            if input.dim == 3
              raise ArgumentError, "3D tensors expect 2 values for padding" unless pad.size == 2
              case mode
              when "reflect"
                NN.reflection_pad1d(input, pad)
              when "replicate"
                NN.replication_pad1d(input, pad)
              else
                raise NotImplementedYet
              end
            elsif input.dim == 4
              raise ArgumentError, "4D tensors expect 4 values for padding" unless pad.size == 4
              case mode
              when "reflect"
                NN.reflection_pad2d(input, pad)
              when "replicate"
                NN.replication_pad2d(input, pad)
              else
                raise NotImplementedYet
              end
            elsif input.dim == 5
              raise ArgumentError, "5D tensors expect 6 values for padding" unless pad.size == 6
              case mode
              when "replicate"
                NN.replication_pad3d(input, pad)
              else
                raise NotImplementedYet
              end
            else
              raise ArgumentError, "Only 3D, 4D, 5D padding with non-constant padding are supported for now"
            end
          end
        end

        # activation layers

        def elu(input, alpha: 1, inplace: false)
          if inplace
            NN.elu!(input, alpha)
          else
            NN.elu(input, alpha)
          end
        end

        def gelu(input, approximate: 'none')
          NN.gelu(input, approximate: approximate)
        end

        def hardshrink(input, lambd = 0.5)
          Torch.hardshrink(input, lambd)
        end

        def leaky_relu(input, negative_slope = 0.01, inplace: false)
          if inplace
            NN.leaky_relu!(input, negative_slope)
          else
            NN.leaky_relu(input, negative_slope)
          end
        end

        def log_sigmoid(input)
          NN.log_sigmoid(input)
        end

        def prelu(input, weight)
          Torch.prelu(input, weight)
        end

        def relu(input, inplace: false)
          if inplace
            input.relu!
          else
            input.relu
          end
        end

        def softplus(input, beta: 1, threshold: 20)
          NN.softplus(input, beta, threshold)
        end

        def softshrink(*args, **options)
          NN.softshrink(*args, **options)
        end

        def softsign(input)
          input / (input.abs + 1)
        end

        def tanhshrink(input)
          input - input.tanh
        end

        # other activation layers

        def softmin(input, dim: nil)
          dim ||= softmax_dim(input.dim)
          (-input).softmax(dim)
        end

        def softmax(input, dim: nil)
          dim ||= softmax_dim(input.dim)
          input.softmax(dim)
        end

        # TODO make dim keyword argument and update examples
        def log_softmax(input, dim = nil)
          dim ||= softmax_dim(input.dim)
          input.log_softmax(dim)
        end

        # normalization layers

        def batch_norm(input, running_mean, running_var, weight: nil, bias: nil,
          training: false, momentum: 0.1, eps: 1e-5)

          if training
            size = input.size
            size_prods = size[0]
            (size.length - 2).times do |i|
              size_prods *= size[i + 2]
            end
            if size_prods == 1
              raise ArgumentError, "Expected more than 1 value per channel when training, got input size #{size.inspect}"
            end
          end

          Torch.batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps, false
          )
        end

        def group_norm(input, num_groups, weight: nil, bias: nil, eps: 1e-5)
          Torch.group_norm(input, num_groups, weight, bias, eps, false)
        end

        def instance_norm(input, running_mean: nil, running_var: nil, weight: nil,
          bias: nil, use_input_stats: true, momentum: 0.1, eps: 1e-5)

          Torch.instance_norm(
              input, weight, bias, running_mean, running_var,
              use_input_stats, momentum, eps, false
          )
        end

        def layer_norm(input, normalized_shape, weight: nil, bias: nil, eps: 1e-5)
          Torch.layer_norm(input, normalized_shape, weight, bias, eps, false)
        end

        def local_response_norm(input, size, alpha: 1e-4, beta: 0.75, k: 1.0)
          dim = input.dim
          if dim < 3
            raise ArgumentError, "Expected 3D or higher dimensionality input (got #{dim} dimensions)"
          end
          div = input.mul(input).unsqueeze(1)
          if dim == 3
            div = pad(div, [0, 0, size / 2, (size - 1) / 2])
            div = avg_pool2d(div, [size, 1], stride: 1).squeeze(1)
          else
            sizes = input.size
            div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
            div = pad(div, [0, 0, 0, 0, size / 2, (size - 1) / 2])
            div = avg_pool3d(div, [size, 1, 1], stride: 1).squeeze(1)
            div = div.view(sizes)
          end
          div = div.mul(alpha).add(k).pow(beta)
          input / div
        end

        # linear layers

        def linear(input, weight, bias)
          NN.linear(input, weight, bias)
        end

        def bilinear(input1, input2, weight, bias)
          Torch.bilinear(input1, input2, weight, bias)
        end

        # dropout layers

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

        # sparse layers

        def embedding(input, weight, padding_idx: nil, max_norm: nil, norm_type: 2.0, scale_grad_by_freq: false, sparse: false)
          # TODO handle max_norm and norm_type
          raise NotImplementedYet unless max_norm.nil? && norm_type == 2.0

          padding_idx ||= -1
          # weight and indices are swapped from Python interface
          Torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
        end

        def embedding_bag(input, weight, offsets: nil, max_norm: nil, norm_type: 2, scale_grad_by_freq: false, mode: "mean", sparse: false, per_sample_weights: nil)
          # TODO handle max_norm and norm_type
          raise NotImplementedYet unless max_norm.nil? && norm_type == 2.0

          mode_enum =
            case mode
            when "sum"
              0
            when "mean"
              1
            when "max"
              2
            else
              raise ArgumentError, "Unknown mode: #{mode}"
            end

          # weight and input swapped
          ret, _, _, _ = Torch.embedding_bag(weight, input, offsets, scale_grad_by_freq, mode_enum, sparse, per_sample_weights)
          ret
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
          NN.binary_cross_entropy(input, target, weight, to_reduction(reduction))
        end

        def binary_cross_entropy_with_logits(input, target, weight: nil, reduction: "mean", pos_weight: nil)
          Torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, to_reduction(reduction))
        end

        def cosine_embedding_loss(input1, input2, target, margin: 0, reduction: "mean")
          Torch.cosine_embedding_loss(input1, input2, target, margin, to_reduction(reduction))
        end

        def cross_entropy(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          nll_loss(log_softmax(input, 1), target, weight: weight, ignore_index: ignore_index, reduction: reduction)
        end

        def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank: 0, reduction: "mean", zero_infinity: false)
          # call to_a on input_lengths and target_lengths for C++
          Torch.ctc_loss(log_probs, targets, input_lengths.to_a, target_lengths.to_a, blank, to_reduction(reduction), zero_infinity)
        end

        def hinge_embedding_loss(input, target, margin: 1.0, reduction: "mean")
          Torch.hinge_embedding_loss(input, target, margin, to_reduction(reduction))
        end

        def kl_div(input, target, reduction: "mean")
          Torch.kl_div(input, target, to_reduction(reduction))
        end

        def l1_loss(input, target, reduction: "mean")
          NN.l1_loss(input, target, to_reduction(reduction))
        end

        def margin_ranking_loss(input1, input2, target, margin: 0, reduction: "mean")
          Torch.margin_ranking_loss(input1, input2, target, margin, to_reduction(reduction))
        end

        def mse_loss(input, target, reduction: "mean")
          if target.size != input.size
            warn "Using a target size (#{target.size}) that is different to the input size (#{input.size}). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size."
          end
          NN.mse_loss(input, target, to_reduction(reduction))
        end

        def multilabel_margin_loss(input, target, reduction: "mean")
          NN.multilabel_margin_loss(input, target, to_reduction(reduction))
        end

        def multilabel_soft_margin_loss(input, target, weight: nil)
          raise NotImplementedYet
        end

        def multi_margin_loss(input, target, p: 1, margin: 1.0, weight: nil, reduction: "mean")
          NN.multi_margin_loss(input, target, p, margin, weight, to_reduction(reduction))
        end

        def nll_loss(input, target, weight: nil, ignore_index: -100, reduction: "mean")
          NN.nll_loss(input, target, weight, to_reduction(reduction), ignore_index)
        end

        def poisson_nll_loss(input, target, log_input: true, full: false, eps: 1e-8, reduction: "mean")
          Torch.poisson_nll_loss(input, target, log_input, full, eps, to_reduction(reduction))
        end

        def soft_margin_loss(input, target, reduction: "mean")
          NN.soft_margin_loss(input, target, to_reduction(reduction))
        end

        def smooth_l1_loss(input, target, reduction: "mean")
          NN.smooth_l1_loss(input, target, to_reduction(reduction))
        end

        def triplet_margin_loss(anchor, positive, negative, margin: 1.0, p: 2, eps: 1e-06, swap: false, reduction: "mean")
          Torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, to_reduction(reduction))
        end

        # vision

        def interpolate(input, size: nil, scale_factor: nil, mode: "nearest", align_corners: nil, recompute_scale_factor: nil)
          if ["nearest", "area"].include?(mode)
            unless align_corners.nil?
              raise ArgumentError, "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
            end
          else
            if align_corners.nil?
              align_corners = false
            end
          end

          scale_factor_len = input.dim - 2
          scale_factor_list = [nil] * scale_factor_len
          # default value of recompute_scale_factor is False
          if !scale_factor.nil? && (recompute_scale_factor == false || recompute_scale_factor.nil?)
            if scale_factor.is_a?(Array)
              _scale_factor_repeated = scale_factor
            else
              _scale_factor_repeated = [scale_factor] * scale_factor_len
            end
            scale_factor_list = _scale_factor_repeated
          end

          # Give this variable a short name because it has to be repeated multiple times below.
          sfl = scale_factor_list

          closed_over_args = [input, size, scale_factor, recompute_scale_factor]
          output_size = _interp_output_size(closed_over_args)
          if input.dim == 3 && mode == "nearest"
            NN.upsample_nearest1d(input, output_size, sfl[0])
          elsif input.dim == 4 && mode == "nearest"
            NN.upsample_nearest2d(input, output_size, sfl[0], sfl[1])
          elsif input.dim == 5 && mode == "nearest"
            NN.upsample_nearest3d(input, output_size, sfl[0], sfl[1], sfl[2])
          elsif input.dim == 3 && mode == "area"
            adaptive_avg_pool1d(input, output_size)
          elsif input.dim == 4 && mode == "area"
            adaptive_avg_pool2d(input, output_size)
          elsif input.dim == 5 && mode == "area"
            adaptive_avg_pool3d(input, output_size)
          elsif input.dim == 3 && mode == "linear"
            # assert align_corners is not None
            NN.upsample_linear1d(input, output_size, align_corners, sfl[0])
          elsif input.dim == 3 && mode == "bilinear"
            raise ArgumentError, "Got 3D input, but bilinear mode needs 4D input"
          elsif input.dim == 3 && mode == "trilinear"
            raise ArgumentError, "Got 3D input, but trilinear mode needs 5D input"
          elsif input.dim == 4 && mode == "linear"
            raise ArgumentError, "Got 4D input, but linear mode needs 3D input"
          elsif input.dim == 4 && mode == "bilinear"
            # assert align_corners is not None
            NN.upsample_bilinear2d(input, output_size, align_corners, sfl[0], sfl[1])
          elsif input.dim == 4 && mode == "trilinear"
            raise ArgumentError, "Got 4D input, but trilinear mode needs 5D input"
          elsif input.dim == 5 && mode == "linear"
            raise ArgumentError, "Got 5D input, but linear mode needs 3D input"
          elsif input.dim == 5 && mode == "bilinear"
            raise ArgumentError, "Got 5D input, but bilinear mode needs 4D input"
          elsif input.dim == 5 && mode == "trilinear"
            # assert align_corners is not None
            NN.upsample_trilinear3d(input, output_size, align_corners, sfl[0], sfl[1], sfl[2])
          elsif input.dim == 4 && mode == "bicubic"
            # assert align_corners is not None
            NN.upsample_bicubic2d(input, output_size, align_corners, sfl[0], sfl[1])
          else
            raise ArgumentError, "Input Error: Only 3D, 4D and 5D input Tensors supported (got #{input.dim}D) for the modes: nearest | linear | bilinear | bicubic | trilinear (got #{mode})"
          end
        end

        private

        # see _reduction.py
        def to_reduction(v)
          case v.to_s
          when "none"
            0
          when "mean"
            1
          when "sum"
            2
          else
            raise ArgumentError, "#{v} is not a valid value for reduction"
          end
        end

        def softmax_dim(ndim)
          ndim == 0 || ndim == 1 || ndim == 3 ? 0 : 1
        end

        def list_with_default(out_size, defaults)
          if out_size.is_a?(Integer)
            out_size
          elsif defaults.length < out_size.length
            raise ArgumentError, "Input dimension should be at least #{out_size.length + 1}"
          else
            out_size.zip(defaults.last(out_size.length)).map { |v, d| v || d }
          end
        end

        def _interp_output_size(closed_over_args)
          input, size, scale_factor, _recompute_scale_factor = closed_over_args
          dim = input.dim - 2
          if size.nil? && scale_factor.nil?
            raise ArgumentError, "either size or scale_factor should be defined"
          end
          if !size.nil? && !scale_factor.nil?
            raise ArgumentError, "only one of size or scale_factor should be defined"
          end
          if !scale_factor.nil?
            if scale_factor.is_a?(Array)
              if scale_factor.length != dim
                raise ArgumentError, "scale_factor shape must match input shape. Input is #{dim}D, scale_factor size is #{scale_factor.length}"
              end
            end
          end

          if !size.nil?
            if size.is_a?(Array)
              return size
            else
              return [size] * dim
            end
          end

          raise "Failed assertion" if scale_factor.nil?
          if scale_factor.is_a?(Array)
            scale_factors = scale_factor
          else
            scale_factors = [scale_factor] * dim
          end

          dim.times.map { |i| (input.size(i + 2) * scale_factors[i]).floor }
        end
      end
    end

    # shortcut
    F = Functional
  end
end
