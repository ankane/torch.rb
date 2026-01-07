# ext
require "torch/ext"

# stdlib
require "fileutils"
require "net/http"
require "set"
require "tmpdir"

# modules
require_relative "torch/device"
require_relative "torch/inspector"
require_relative "torch/tensor"
require_relative "torch/version"

# distributions
require_relative "torch/distributions/distribution"
require_relative "torch/distributions/exponential_family"
require_relative "torch/distributions/normal"
require_relative "torch/distributions/utils"

# optim
require_relative "torch/optim/optimizer"
require_relative "torch/optim/adadelta"
require_relative "torch/optim/adagrad"
require_relative "torch/optim/adam"
require_relative "torch/optim/adamax"
require_relative "torch/optim/adamw"
require_relative "torch/optim/asgd"
require_relative "torch/optim/rmsprop"
require_relative "torch/optim/rprop"
require_relative "torch/optim/sgd"

# optim lr_scheduler
require_relative "torch/optim/lr_scheduler/lr_scheduler"
require_relative "torch/optim/lr_scheduler/lambda_lr"
require_relative "torch/optim/lr_scheduler/multiplicative_lr"
require_relative "torch/optim/lr_scheduler/step_lr"
require_relative "torch/optim/lr_scheduler/multi_step_lr"
require_relative "torch/optim/lr_scheduler/exponential_lr"
require_relative "torch/optim/lr_scheduler/cosine_annealing_lr"

# nn parameters
require_relative "torch/nn/parameter"
require_relative "torch/nn/utils"

# nn containers
require_relative "torch/nn/module"
require_relative "torch/nn/module_list"
require_relative "torch/nn/parameter_list"
require_relative "torch/nn/sequential"

# nn convolution layers
require_relative "torch/nn/convnd"
require_relative "torch/nn/conv1d"
require_relative "torch/nn/conv2d"
require_relative "torch/nn/conv3d"
require_relative "torch/nn/unfold"
require_relative "torch/nn/fold"

# nn pooling layers
require_relative "torch/nn/max_poolnd"
require_relative "torch/nn/max_pool1d"
require_relative "torch/nn/max_pool2d"
require_relative "torch/nn/max_pool3d"
require_relative "torch/nn/max_unpoolnd"
require_relative "torch/nn/max_unpool1d"
require_relative "torch/nn/max_unpool2d"
require_relative "torch/nn/max_unpool3d"
require_relative "torch/nn/avg_poolnd"
require_relative "torch/nn/avg_pool1d"
require_relative "torch/nn/avg_pool2d"
require_relative "torch/nn/avg_pool3d"
require_relative "torch/nn/lp_poolnd"
require_relative "torch/nn/lp_pool1d"
require_relative "torch/nn/lp_pool2d"
require_relative "torch/nn/adaptive_max_poolnd"
require_relative "torch/nn/adaptive_max_pool1d"
require_relative "torch/nn/adaptive_max_pool2d"
require_relative "torch/nn/adaptive_max_pool3d"
require_relative "torch/nn/adaptive_avg_poolnd"
require_relative "torch/nn/adaptive_avg_pool1d"
require_relative "torch/nn/adaptive_avg_pool2d"
require_relative "torch/nn/adaptive_avg_pool3d"

# nn padding layers
require_relative "torch/nn/reflection_padnd"
require_relative "torch/nn/reflection_pad1d"
require_relative "torch/nn/reflection_pad2d"
require_relative "torch/nn/replication_padnd"
require_relative "torch/nn/replication_pad1d"
require_relative "torch/nn/replication_pad2d"
require_relative "torch/nn/replication_pad3d"
require_relative "torch/nn/constant_padnd"
require_relative "torch/nn/constant_pad1d"
require_relative "torch/nn/constant_pad2d"
require_relative "torch/nn/constant_pad3d"
require_relative "torch/nn/zero_pad2d"

# nn normalization layers
require_relative "torch/nn/batch_norm"
require_relative "torch/nn/batch_norm1d"
require_relative "torch/nn/batch_norm2d"
require_relative "torch/nn/batch_norm3d"
require_relative "torch/nn/group_norm"
require_relative "torch/nn/instance_norm"
require_relative "torch/nn/instance_norm1d"
require_relative "torch/nn/instance_norm2d"
require_relative "torch/nn/instance_norm3d"
require_relative "torch/nn/layer_norm"
require_relative "torch/nn/local_response_norm"

# nn recurrent layers
require_relative "torch/nn/rnn_base"
require_relative "torch/nn/rnn"
require_relative "torch/nn/lstm"
require_relative "torch/nn/gru"

# nn linear layers
require_relative "torch/nn/bilinear"
require_relative "torch/nn/identity"
require_relative "torch/nn/linear"

# nn dropout layers
require_relative "torch/nn/dropoutnd"
require_relative "torch/nn/alpha_dropout"
require_relative "torch/nn/dropout"
require_relative "torch/nn/dropout2d"
require_relative "torch/nn/dropout3d"
require_relative "torch/nn/feature_alpha_dropout"

# nn activations
require_relative "torch/nn/elu"
require_relative "torch/nn/gelu"
require_relative "torch/nn/hardshrink"
require_relative "torch/nn/leaky_relu"
require_relative "torch/nn/log_sigmoid"
require_relative "torch/nn/prelu"
require_relative "torch/nn/relu"
require_relative "torch/nn/sigmoid"
require_relative "torch/nn/softplus"
require_relative "torch/nn/softshrink"
require_relative "torch/nn/softsign"
require_relative "torch/nn/tanh"
require_relative "torch/nn/tanhshrink"

# nn activations other
require_relative "torch/nn/log_softmax"
require_relative "torch/nn/softmax"
require_relative "torch/nn/softmax2d"
require_relative "torch/nn/softmin"

# nn sparse layers
require_relative "torch/nn/embedding"
require_relative "torch/nn/embedding_bag"

# attention is all you need
require_relative "torch/nn/multihead_attention"
require_relative "torch/nn/transformer"

# nn distance functions
require_relative "torch/nn/cosine_similarity"
require_relative "torch/nn/pairwise_distance"

# nn loss functions
require_relative "torch/nn/loss"
require_relative "torch/nn/weighted_loss"
require_relative "torch/nn/bce_loss"
require_relative "torch/nn/bce_with_logits_loss"
require_relative "torch/nn/cosine_embedding_loss"
require_relative "torch/nn/cross_entropy_loss"
require_relative "torch/nn/ctc_loss"
require_relative "torch/nn/hinge_embedding_loss"
require_relative "torch/nn/kl_div_loss"
require_relative "torch/nn/l1_loss"
require_relative "torch/nn/margin_ranking_loss"
require_relative "torch/nn/mse_loss"
require_relative "torch/nn/multi_label_margin_loss"
require_relative "torch/nn/multi_label_soft_margin_loss"
require_relative "torch/nn/multi_margin_loss"
require_relative "torch/nn/nll_loss"
require_relative "torch/nn/poisson_nll_loss"
require_relative "torch/nn/smooth_l1_loss"
require_relative "torch/nn/soft_margin_loss"
require_relative "torch/nn/triplet_margin_loss"

# nn vision
require_relative "torch/nn/upsample"

# nn other
require_relative "torch/nn/functional"
require_relative "torch/nn/functional_attention"
require_relative "torch/nn/init"

# utils
require_relative "torch/utils/data"
require_relative "torch/utils/data/data_loader"
require_relative "torch/utils/data/dataset"
require_relative "torch/utils/data/iterable_dataset"
require_relative "torch/utils/data/data_pipes/iter_data_pipe"
require_relative "torch/utils/data/data_pipes/filter_iter_data_pipe"
require_relative "torch/utils/data/data_pipes/iter/file_lister"
require_relative "torch/utils/data/data_pipes/iter/file_opener"
require_relative "torch/utils/data/data_pipes/iter/iterable_wrapper"
require_relative "torch/utils/data/data_pipes/iter/stream_wrapper"
require_relative "torch/utils/data/subset"
require_relative "torch/utils/data/tensor_dataset"

# hub
require_relative "torch/hub"

module Torch
  class NotImplementedYet < StandardError
    def message
      "This feature has not been implemented yet. Consider submitting a PR."
    end
  end

  # legacy
  # but may make it easier to port tutorials
  module Autograd
    class Variable
      def self.new(x)
        raise ArgumentError, "Variable data has to be a tensor, but got #{x.class.name}" unless x.is_a?(Tensor)
        warn "[torch] The Variable API is deprecated. Use tensors with requires_grad: true instead."
        x
      end
    end
  end

  # TODO move to C++
  class ByteStorage
    # private
    attr_reader :bytes

    def initialize(bytes)
      @bytes = bytes
    end

    def self.from_buffer(bytes)
      new(bytes)
    end
  end

  # keys: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
  # values: https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
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
    complex_half: 8,
    complex32: 8,
    complex_float: 9,
    complex64: 9,
    complex_double: 10,
    cdouble: 10,
    complex128: 10,
    bool: 11,
    qint8: 12,
    quint8: 13,
    qint32: 14,
    bfloat16: 15
  }
  ENUM_TO_DTYPE = DTYPE_TO_ENUM.map(&:reverse).to_h

  TENSOR_TYPE_CLASSES = []

  def self._make_tensor_class(dtype, cuda = false)
    cls = Class.new
    device = cuda ? "cuda" : "cpu"
    cls.define_singleton_method("new") do |*args|
      if args.size == 1 && args.first.is_a?(Tensor)
        args.first.send(dtype).to(device)
      elsif args.size == 1 && args.first.is_a?(ByteStorage) && dtype == :uint8
        bytes = args.first.bytes
        Torch._from_blob_ref(bytes, [bytes.bytesize], TensorOptions.new.dtype(DTYPE_TO_ENUM[dtype]))
      elsif args.size == 1 && args.first.is_a?(Array)
        Torch.tensor(args.first, dtype: dtype, device: device)
      elsif args.size == 0
        Torch.empty(0, dtype: dtype, device: device)
      else
        Torch.empty(*args, dtype: dtype, device: device)
      end
    end
    TENSOR_TYPE_CLASSES << cls
    cls
  end

  DTYPE_TO_CLASS = {
    float32: "FloatTensor",
    float64: "DoubleTensor",
    float16: "HalfTensor",
    uint8: "ByteTensor",
    int8: "CharTensor",
    int16: "ShortTensor",
    int32: "IntTensor",
    int64: "LongTensor",
    bool: "BoolTensor"
  }

  DTYPE_TO_CLASS.each do |dtype, class_name|
    const_set(class_name, _make_tensor_class(dtype))
    CUDA.const_set(class_name, _make_tensor_class(dtype, true))
  end

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
      _from_blob_ref(ndarray.to_string, ndarray.shape, options)
    end

    # private
    # TODO use keepAlive in Rice (currently segfaults)
    def _from_blob_ref(data, size, options)
      tensor = _from_blob(data, size, options)
      # from_blob does not own the data, so we need to keep
      # a reference to it for duration of tensor
      # can remove when passing pointer directly
      tensor.instance_variable_set("@_numo_data", data)
      tensor
    end

    # private
    # use method for cases when Numo not available
    # or available after Torch loaded
    def _dtype_to_numo
      raise Error, "Numo not found" unless defined?(Numo::NArray)

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

    def no_grad(&block)
      grad_enabled(false, &block)
    end

    def enable_grad(&block)
      grad_enabled(true, &block)
    end

    def grad_enabled(value)
      previous_value = grad_enabled?
      begin
        _set_grad_enabled(value)
        yield
      ensure
        _set_grad_enabled(previous_value)
      end
    end
    alias_method :set_grad_enabled, :grad_enabled

    def device(str)
      if str.is_a?(Device)
        str
      else
        Device.new(str)
      end
    end

    def save(obj, f)
      File.binwrite(f, _save(to_ivalue(obj)))
    end

    def load(filename, map_location: nil, weights_only: false)
      # keep backwards compatibility
      File.open(filename, "rb") { |f| f.read(1) }

      load_device = map_location_device(map_location) if map_location
      result =
        if load_device
          device_str =
            if load_device.respond_to?(:_str)
              load_device._str
            else
              load_device.to_s
            end
          to_ruby(_load_with_device(filename, device_str))
        else
          to_ruby(_load(filename))
        end
      ensure_weights_only_contents!(result) if weights_only
      result = apply_map_location(result, map_location) if map_location
      result
    end

    def tensor(data, **options)
      if options[:dtype].nil? && defined?(Numo::NArray) && data.is_a?(Numo::NArray)
        numo_to_dtype = _dtype_to_numo.map(&:reverse).to_h
        options[:dtype] = numo_to_dtype[data.class]
      end

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

      if options[:dtype].nil?
        if data.all? { |v| v.is_a?(Integer) }
          options[:dtype] = :int64
        elsif data.all? { |v| v == true || v == false }
          options[:dtype] = :bool
        elsif data.any? { |v| v.is_a?(Complex) }
          options[:dtype] = :complex64
        end
      end

      # TODO check each dimensions for consistency in future
      raise Error, "Inconsistent dimensions" if data.size != size.inject(1, :*)

      # TODO move to C++
      data = data.map { |v| v ? 1 : 0 } if options[:dtype] == :bool

      _tensor(data, size, tensor_options(**options))
    end

    private

    def to_ivalue(obj)
      case obj
      when String
        IValue.from_string(obj)
      when Integer
        IValue.from_int(obj)
      when Tensor
        IValue.from_tensor(obj)
      when Float
        IValue.from_double(obj)
      when Hash
        dict = {}
        obj.each do |k, v|
          dict[to_ivalue(k)] = to_ivalue(v)
        end
        IValue.from_dict(dict)
      when true, false
        IValue.from_bool(obj)
      when nil
        IValue.new
      when Array
        IValue.from_list(obj.map { |v| to_ivalue(v) })
      else
        raise Error, "Unknown type: #{obj.class.name}"
      end
    end

    def to_ruby(ivalue)
      if ivalue.bool?
        ivalue.to_bool
      elsif ivalue.double?
        ivalue.to_double
      elsif ivalue.int?
        ivalue.to_int
      elsif ivalue.none?
        nil
      elsif ivalue.string?
        ivalue.to_string_ref
      elsif ivalue.tensor?
        ivalue.to_tensor
      elsif ivalue.generic_dict?
        dict = {}
        ivalue.to_generic_dict.each do |k, v|
          dict[to_ruby(k)] = to_ruby(v)
        end
        dict
      elsif ivalue.list?
        ivalue.to_list.map { |v| to_ruby(v) }
      else
        type =
          if ivalue.capsule?
            "Capsule"
          elsif ivalue.custom_class?
            "CustomClass"
          elsif ivalue.tuple?
            "Tuple"
          elsif ivalue.future?
            "Future"
          elsif ivalue.r_ref?
            "RRef"
          elsif ivalue.int_list?
            "IntList"
          elsif ivalue.double_list?
            "DoubleList"
          elsif ivalue.bool_list?
            "BoolList"
          elsif ivalue.tensor_list?
            "TensorList"
          elsif ivalue.object?
            "Object"
          elsif ivalue.module?
            "Module"
          elsif ivalue.py_object?
            "PyObject"
          elsif ivalue.scalar?
            "Scalar"
          elsif ivalue.device?
            "Device"
          # elsif ivalue.generator?
          #   "Generator"
          elsif ivalue.ptr_type?
            "PtrType"
          else
            "Unknown"
          end

        raise Error, "Unsupported type: #{type}"
      end
    end

    WEIGHTS_ONLY_PRIMITIVE_CLASSES =
      [
        NilClass,
        TrueClass,
        FalseClass,
        Integer,
        Float,
        String
      ].freeze

    def ensure_weights_only_contents!(obj)
      case obj
      when *WEIGHTS_ONLY_PRIMITIVE_CLASSES
        obj
      when Tensor
        obj
      when Array
        obj.each { |value| ensure_weights_only_contents!(value) }
      when Hash
        obj.each do |key, value|
          ensure_weights_only_contents!(key)
          ensure_weights_only_contents!(value)
        end
      else
        raise Error, "weights_only load supports tensors, primitive Ruby types, arrays, and hashes (found #{obj.class.name})"
      end
    end

    def map_location_device(map_location)
      case map_location
      when Device, String, Symbol
        normalize_map_location_device(map_location)
      when Hash
        devices =
          map_location.values.map do |value|
            normalize_map_location_device(value)
          rescue StandardError
            nil
          end.compact
        return nil if devices.empty?
        devices.uniq!
        devices.one? ? devices.first : nil
      else
        nil
      end
    end

    def apply_map_location(obj, map_location)
      case obj
      when Tensor
        map_tensor_location(obj, map_location)
      when Array
        obj.map { |value| apply_map_location(value, map_location) }
      when Hash
        obj.each_with_object({}) do |(key, value), memo|
          memo[apply_map_location(key, map_location)] = apply_map_location(value, map_location)
        end
      else
        obj
      end
    end

    def map_tensor_location(tensor, map_location)
      case map_location
      when nil
        tensor
      when Hash
        target = lookup_map_location_target(map_location, tensor.device)
        return tensor if target.nil?
        map_tensor_location(tensor, target)
      else
        return map_tensor_location_callable(tensor, map_location) if map_location.respond_to?(:call)
        device = normalize_map_location_device(map_location)
        tensor.to(device)
      end
    end

    def map_tensor_location_callable(tensor, callable)
      mapped = callable.call(tensor, map_location_device_tag(tensor.device))
      return tensor if mapped.nil?
      unless mapped.is_a?(Tensor)
        raise Error, "map_location callable must return a Tensor or nil (got #{mapped.class.name})"
      end
      mapped
    end

    def lookup_map_location_target(mapping, device)
      key = map_location_device_tag(device)
      mapping.each do |candidate, value|
        candidate_key =
          case candidate
          when Device
            map_location_device_tag(candidate)
          when String, Symbol
            candidate.to_s
          else
            candidate
          end
        return value if candidate_key == key
      end
      nil
    end

    def map_location_device_tag(device)
      case device
      when Device
        tag = device.type
        tag += ":#{device.index}" unless device.index.nil?
        tag
      when String, Symbol
        device.to_s
      else
        raise Error, "Unknown device reference: #{device.inspect}"
      end
    end

    def normalize_map_location_device(location)
      case location
      when Device
        location
      when String, Symbol
        device(location.to_s)
      else
        raise Error, "Unsupported map_location: #{location.inspect}"
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
  end
end
