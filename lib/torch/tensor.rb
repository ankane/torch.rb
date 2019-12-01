module Torch
  class Tensor
    include Comparable
    include Inspector

    alias_method :requires_grad?, :requires_grad

    def self.new(*size)
      if size.length == 1 && size.first.is_a?(Tensor)
        size.first
      else
        Torch.empty(*size)
      end
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

    def to_a
      reshape_arr(_data, shape)
    end

    # TODO support dtype
    def to(device, non_blocking: false, copy: false)
      device = Device.new(device) if device.is_a?(String)
      _to(device, _dtype, non_blocking, copy)
    end

    def size(dim = nil)
      if dim
        _size(dim)
      else
        shape
      end
    end

    def shape
      dim.times.map { |i| size(i) }
    end

    def view(*size)
      _view(size)
    end

    def item
      if numel != 1
        raise Error, "only one element tensors can be converted to Ruby scalars"
      end
      _data.first
    end

    # unsure if this is correct
    def new
      Torch.empty(0, dtype: dtype)
    end

    def backward(gradient = nil)
      _backward(gradient)
    end

    # TODO read directly from memory
    def numo
      cls = Torch._dtype_to_numo[dtype]
      raise Error, "Cannot convert #{dtype} to Numo" unless cls
      cls.cast(_data).reshape(*shape)
    end

    def new_ones(*size, **options)
      Torch.ones_like(Torch.empty(*size), **options)
    end

    def requires_grad!(requires_grad = true)
      _requires_grad!(requires_grad)
    end

    def type(dtype)
      enum = DTYPE_TO_ENUM[dtype]
      raise Error, "Unknown type: #{dtype}" unless enum
      _type(enum)
    end

    def add!(value = 1, other)
      if other.is_a?(Numeric)
        _add_scalar!(other * value)
      else
        # need to use alpha for sparse tensors instead of multiplying
        _add_alpha!(other, value)
      end
    end

    def mul!(other)
      if other.is_a?(Numeric)
        _mul_scalar!(other)
      else
        _mul!(other)
      end
    end

    # operations
    %w(abs add argmax div dot eq exp gt log log_softmax lt matmul max mean min mul neg norm num numel pow relu remainder reshape sign softmax sqrt sub sum unsqueeze topk).each do |op|
      define_method(op) do |*args, **options, &block|
        if options.any?
          Torch.send(op, self, *args, **options, &block)
        else
          Torch.send(op, self, *args, &block)
        end
      end
    end

    def +(other)
      add(other)
    end

    def -(other)
      sub(other)
    end

    def *(other)
      mul(other)
    end

    def /(other)
      div(other)
    end

    def %(other)
      remainder(other)
    end

    def **(other)
      pow(other)
    end

    def -@
      neg
    end

    def <=>(other)
      item <=> other
    end

    # based on python_variable_indexing.cpp
    def [](*indexes)
      result = self
      dim = 0
      indexes.each do |index|
        if index.is_a?(Numeric)
          result = result._select(dim, index)
        elsif index.is_a?(Range)
          finish = index.end
          finish += 1 unless index.exclude_end?
          result = result._slice(dim, index.begin, finish, 1)
          dim += 1
        elsif index.nil?
          result = result.unsqueeze(dim)
          dim += 1
        elsif index == true
          result = result.unsqueeze(dim)
          # TODO handle false
        else
          raise Error, "Unsupported index type: #{index.class.name}"
        end
      end
      result
    end

    # TODO
    # based on python_variable_indexing.cpp
    def []=(index, value)
      raise ArgumentError, "Tensor does not support deleting items" if value.nil?

      value = Torch.tensor(value) unless value.is_a?(Tensor)

      if index.is_a?(Numeric)
        copy_to(_select(0, index), value)
      elsif index.is_a?(Range)
        finish = index.end
        finish += 1 unless index.exclude_end?
        copy_to(_slice(0, index.begin, finish, 1), value)
      else
        raise Error, "Unsupported index type: #{index.class.name}"
      end
    end

    private

    def copy_to(dst, src)
      dst.copy!(src)
    end

    def reshape_arr(arr, dims)
      if dims.empty?
        arr
      else
        arr = arr.flatten
        dims[1..-1].reverse.each do |dim|
          arr = arr.each_slice(dim)
        end
        arr.to_a
      end
    end
  end
end
