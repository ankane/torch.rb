module Torch
  class Tensor
    include Comparable
    include Enumerable
    include Inspector

    alias_method :requires_grad?, :requires_grad
    alias_method :ndim, :dim
    alias_method :ndimension, :dim

    # use alias_method for performance
    alias_method :+, :add
    alias_method :-, :sub
    alias_method :*, :mul
    alias_method :/, :div
    alias_method :%, :remainder
    alias_method :**, :pow
    alias_method :-@, :neg
    alias_method :&, :logical_and
    alias_method :|, :logical_or
    alias_method :^, :logical_xor

    def self.new(*args)
      FloatTensor.new(*args)
    end

    def dtype
      dtype = ENUM_TO_DTYPE[_dtype]
      raise Error, "Unknown type: #{_dtype}" unless dtype
      dtype
    end

    def layout
      _layout.downcase.to_sym
    end

    def to_s
      inspect
    end

    def each
      return enum_for(:each) unless block_given?

      size(0).times do |i|
        yield self[i]
      end
    end

    # TODO make more performant
    def to_a
      arr = _flat_data
      if shape.empty?
        arr
      else
        shape[1..-1].reverse.each do |dim|
          arr = arr.each_slice(dim)
        end
        arr.to_a
      end
    end

    def to(device = nil, dtype: nil, non_blocking: false, copy: false)
      if device.is_a?(Symbol) && !dtype
        dtype = device
        device = nil
      end

      device ||= self.device
      device = Device.new(device) if device.is_a?(String)

      dtype ||= self.dtype
      enum = DTYPE_TO_ENUM[dtype]
      raise Error, "Unknown type: #{dtype}" unless enum

      _to(device, enum, non_blocking, copy)
    end

    def cpu
      to("cpu")
    end

    def cuda
      to("cuda")
    end

    def size(dim = nil)
      if dim
        _size(dim)
      else
        shape
      end
    end

    def stride(dim = nil)
      if dim
        _stride(dim)
      else
        _strides
      end
    end

    # mirror Python len()
    def length
      size(0)
    end

    def item
      if numel != 1
        raise Error, "only one element tensors can be converted to Ruby scalars"
      end
      to_a.first
    end

    def to_i
      item.to_i
    end

    def to_f
      item.to_f
    end

    # unsure if this is correct
    def new
      Torch.empty(0, dtype: dtype)
    end

    # TODO read directly from memory
    def numo
      cls = Torch._dtype_to_numo[dtype]
      raise Error, "Cannot convert #{dtype} to Numo" unless cls
      cls.from_string(_data_str).reshape(*shape)
    end

    def new_ones(*size, **options)
      Torch.ones_like(Torch.empty(*size), **options)
    end

    def requires_grad=(requires_grad)
      _requires_grad!(requires_grad)
    end

    def requires_grad!(requires_grad = true)
      _requires_grad!(requires_grad)
    end

    def type(dtype)
      if dtype.is_a?(Class)
        raise Error, "Invalid type: #{dtype}" unless TENSOR_TYPE_CLASSES.include?(dtype)
        dtype.new(self)
      else
        enum = DTYPE_TO_ENUM[dtype]
        raise Error, "Invalid type: #{dtype}" unless enum
        _type(enum)
      end
    end

    # TODO better compare?
    def <=>(other)
      item <=> other
    end

    # based on python_variable_indexing.cpp and
    # https://pytorch.org/cppdocs/notes/tensor_indexing.html
    def [](*indexes)
      _index(indexes)
    end

    # based on python_variable_indexing.cpp and
    # https://pytorch.org/cppdocs/notes/tensor_indexing.html
    def []=(*indexes, value)
      raise ArgumentError, "Tensor does not support deleting items" if value.nil?
      value = Torch.tensor(value, dtype: dtype) unless value.is_a?(Tensor)
      _index_put_custom(indexes, value)
    end

    # parser can't handle overlap, so need to handle manually
    def random!(*args)
      return _random!(0, *args) if args.size == 1
      _random!(*args)
    end

    # center option
    def stft(*args)
      Torch.stft(*args)
    end
  end
end
