# ext
require "torch/ext"

# modules
require "torch/inspector"
require "torch/tensor"
require "torch/version"

# optim
require "torch/optim/optimizer"
require "torch/optim/adadelta"
require "torch/optim/adagrad"
require "torch/optim/adam"
require "torch/optim/adamax"
require "torch/optim/adamw"
require "torch/optim/asgd"
require "torch/optim/rmsprop"
require "torch/optim/rprop"
require "torch/optim/sgd"

# optim lr_scheduler
require "torch/optim/lr_scheduler/lr_scheduler"
require "torch/optim/lr_scheduler/step_lr"

# nn parameters
require "torch/nn/parameter"

# nn containers
require "torch/nn/module"
require "torch/nn/sequential"

# nn convolution layers
require "torch/nn/convnd"
require "torch/nn/conv2d"

# nn pooling layers
require "torch/nn/max_poolnd"
require "torch/nn/max_pool2d"

# nn linear layers
require "torch/nn/bilinear"
require "torch/nn/identity"
require "torch/nn/linear"

# nn dropout layers
require "torch/nn/dropoutnd"
require "torch/nn/alpha_dropout"
require "torch/nn/dropout"
require "torch/nn/dropout2d"
require "torch/nn/dropout3d"
require "torch/nn/feature_alpha_dropout"

# nn activiations
require "torch/nn/leaky_relu"
require "torch/nn/prelu"
require "torch/nn/relu"
require "torch/nn/sigmoid"
require "torch/nn/softplus"

# nn activations other
require "torch/nn/log_softmax"
require "torch/nn/softmax"
require "torch/nn/softmax2d"
require "torch/nn/softmin"

# nn sparse layers
require "torch/nn/embedding"
require "torch/nn/embedding_bag"

# nn distance functions
require "torch/nn/cosine_similarity"
require "torch/nn/pairwise_distance"

# nn loss functions
require "torch/nn/loss"
require "torch/nn/weighted_loss"
require "torch/nn/bce_loss"
# require "torch/nn/bce_with_logits_loss"
# require "torch/nn/cosine_embedding_loss"
require "torch/nn/cross_entropy_loss"
require "torch/nn/ctc_loss"
# require "torch/nn/hinge_embedding_loss"
require "torch/nn/kl_div_loss"
require "torch/nn/l1_loss"
# require "torch/nn/margin_ranking_loss"
require "torch/nn/mse_loss"
# require "torch/nn/multi_label_margin_loss"
# require "torch/nn/multi_label_soft_margin_loss"
# require "torch/nn/multi_margin_loss"
require "torch/nn/nll_loss"
require "torch/nn/poisson_nll_loss"
# require "torch/nn/smooth_l1_loss"
# require "torch/nn/soft_margin_loss"
# require "torch/nn/triplet_margin_loss"

# nn other
require "torch/nn/functional"
require "torch/nn/init"

# utils
require "torch/utils/data/data_loader"
require "torch/utils/data/tensor_dataset"

module Torch
  class Error < StandardError; end
  class NotImplementedYet < StandardError
    def message
      "This feature has not been implemented yet. Consider submitting a PR."
    end
  end

  # keys: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
  # values: https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
  # complex and quantized types not supported by PyTorch yet
  DTYPE_TO_ENUM = {
    uint8: 0,
    int8: 1,
    short: 2,
    int16: 2,
    int: 3,
    int32: 3,
    long: 4,
    int64: 4,
    half: 5,
    float16: 5,
    float: 6,
    float32: 6,
    double: 7,
    float64: 7,
    # complex_half: 8,
    # complex_float: 9,
    # complex_double: 10,
    bool: 11,
    # qint8: 12,
    # quint8: 13,
    # qint32: 14,
    # bfloat16: 15
  }
  ENUM_TO_DTYPE = DTYPE_TO_ENUM.map(&:reverse).to_h

  class << self
    # Torch.float, Torch.long, etc
    DTYPE_TO_ENUM.each_key do |dtype|
      define_method(dtype) do
        dtype
      end

      Tensor.define_method(dtype) do
        type(dtype)
      end
    end

    # https://pytorch.org/docs/stable/torch.html

    def tensor?(obj)
      obj.is_a?(Tensor)
    end

    def from_numo(ndarray)
      dtype = _dtype_to_numo.find { |k, v| ndarray.is_a?(v) }
      raise Error, "Cannot convert #{ndarray.class.name} to tensor" unless dtype
      options = tensor_options(device: "cpu", dtype: dtype[0])
      # TODO pass pointer to array instead of creating string
      str = ndarray.to_string
      tensor = _from_blob(str, ndarray.shape, options)
      # from_blob does not own the data, so we need to keep
      # a reference to it for duration of tensor
      # can remove when passing pointer directly
      tensor.instance_variable_set("@_numo_str", str)
      tensor
    end

    # private
    # use method for cases when Numo not available
    # or available after Torch loaded
    def _dtype_to_numo
      {
        uint8: Numo::UInt8,
        int8: Numo::Int8,
        int16: Numo::Int16,
        int32: Numo::Int32,
        int64: Numo::Int64,
        float32: Numo::SFloat,
        float64: Numo::DFloat
      }
    end

    # --- begin tensor creation: https://pytorch.org/cppdocs/notes/tensor_creation.html ---

    def arange(start, finish = nil, step = 1, **options)
      # ruby doesn't support start = 0, finish, step = 1, ...
      if finish.nil?
        finish = start
        start = 0
      end
      _arange(start, finish, step, tensor_options(**options))
    end

    def empty(*size, **options)
      _empty(tensor_size(size), tensor_options(**options))
    end

    def eye(n, m = nil, **options)
      _eye(n, m || n, tensor_options(**options))
    end

    def full(size, fill_value, **options)
      _full(size, fill_value, tensor_options(**options))
    end

    def linspace(start, finish, steps = 100, **options)
      _linspace(start, finish, steps, tensor_options(**options))
    end

    def logspace(start, finish, steps = 100, base = 10.0, **options)
      _logspace(start, finish, steps, base, tensor_options(**options))
    end

    def ones(*size, **options)
      _ones(tensor_size(size), tensor_options(**options))
    end

    def rand(*size, **options)
      _rand(tensor_size(size), tensor_options(**options))
    end

    def randint(low = 0, high, size, **options)
      _randint(low, high, size, tensor_options(**options))
    end

     def randn(*size, **options)
      _randn(tensor_size(size), tensor_options(**options))
    end

    def randperm(n, **options)
      _randperm(n, tensor_options(**options))
    end

    def zeros(*size, **options)
      _zeros(tensor_size(size), tensor_options(**options))
    end

    def tensor(data, **options)
      size = []
      if data.respond_to?(:to_a)
        data = data.to_a
        d = data
        while d.is_a?(Array)
          size << d.size
          d = d.first
        end
        data = data.flatten
      else
        data = [data].compact
      end

      if options[:dtype].nil? && data.all? { |v| v.is_a?(Integer) }
        options[:dtype] = :int64
      end

      _tensor(data, size, tensor_options(**options))
    end

    # --- begin like ---

    def ones_like(input, **options)
      ones(input.size, like_options(input, options))
    end

    def empty_like(input, **options)
      empty(input.size, like_options(input, options))
    end

    def full_like(input, fill_value, **options)
      full(input.size, fill_value, like_options(input, options))
    end

    def rand_like(input, **options)
      rand(input.size, like_options(input, options))
    end

    def randint_like(input, low, high = nil, **options)
      # ruby doesn't support input, low = 0, high, ...
      if high.nil?
        high = low
        low = 0
      end
      randint(low, high, input.size, like_options(input, options))
    end

    def randn_like(input, **options)
      randn(input.size, like_options(input, options))
    end

    def zeros_like(input, **options)
      zeros(input.size, like_options(input, options))
    end

    # --- begin operations ---

    %w(add sub mul div remainder).each do |op|
      define_method(op) do |input, other, **options|
        execute_op(op, input, other, **options)
      end
    end

    def neg(input)
      _neg(input)
    end

    def no_grad
      previous_value = grad_enabled?
      begin
        _set_grad_enabled(false)
        yield
      ensure
        _set_grad_enabled(previous_value)
      end
    end

    # TODO support out
    def mean(input, dim = nil, keepdim: false)
      if dim
        _mean_dim(input, dim, keepdim)
      else
        _mean(input)
      end
    end

    # TODO support dtype
    def sum(input, dim = nil, keepdim: false)
      if dim
        _sum_dim(input, dim, keepdim)
      else
        _sum(input)
      end
    end

    def argmax(input, dim = nil, keepdim: false)
      if dim
        _argmax_dim(input, dim, keepdim)
      else
        _argmax(input)
      end
    end

    def eq(input, other)
      _eq(input, other)
    end

    def norm(input)
      _norm(input)
    end

    def pow(input, exponent)
      _pow(input, exponent)
    end

    def min(input)
      _min(input)
    end

    def max(input, dim = nil, keepdim: false, out: nil)
      if dim
        raise NotImplementedYet unless out
        _max_out(out[0], out[1], input, dim, keepdim)
      else
        _max(input)
      end
    end

    def exp(input)
      _exp(input)
    end

    def log(input)
      _log(input)
    end

    def sign(input)
      _sign(input)
    end

    def sigmoid(input)
      _sigmoid(input)
    end

    def gt(input, other)
      _gt(input, other)
    end

    def lt(input, other)
      _lt(input, other)
    end

    def unsqueeze(input, dim)
      _unsqueeze(input, dim)
    end

    def dot(input, tensor)
      _dot(input, tensor)
    end

    def cat(tensors, dim = 0)
      _cat(tensors, dim)
    end

    def matmul(input, other)
      _matmul(input, other)
    end

    def reshape(input, shape)
      _reshape(input, shape)
    end

    def flatten(input, start_dim: 0, end_dim: -1)
      _flatten(input, start_dim, end_dim)
    end

    def sqrt(input)
      _sqrt(input)
    end

    # TODO make dim keyword argument
    def log_softmax(input, dim)
      _log_softmax(input, dim)
    end

    def softmax(input, dim: nil)
      _softmax(input, dim)
    end

    def abs(input)
      _abs(input)
    end

    def device(str)
      Device.new(str)
    end

    private

    def execute_op(op, input, other, out: nil)
      scalar = other.is_a?(Numeric)
      if out
        # TODO make work with scalars
        raise Error, "out not supported with scalar yet" if scalar
        send("_#{op}_out", out, input, other)
      else
        if scalar
          send("_#{op}_scalar", input, other)
        else
          send("_#{op}", input, other)
        end
      end
    end

    def tensor_size(size)
      size.flatten
    end

    def tensor_options(dtype: nil, layout: nil, device: nil, requires_grad: nil)
      options = TensorOptions.new
      unless dtype.nil?
        type = DTYPE_TO_ENUM[dtype]
        raise Error, "Unknown dtype: #{dtype.inspect}" unless type
        options = options.dtype(type)
      end
      unless device.nil?
        options = options.device(device.to_s)
      end
      unless layout.nil?
        options = options.layout(layout.to_s)
      end
      unless requires_grad.nil?
        options = options.requires_grad(requires_grad)
      end
      options
    end

    def like_options(input, options)
      options = options.dup
      options[:dtype] ||= input.dtype
      options[:layout] ||= input.layout
      options[:device] ||= input.device
      options
    end
  end
end
