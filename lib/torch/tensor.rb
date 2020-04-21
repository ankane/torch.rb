module Torch
  class Tensor
    include Comparable
    include Inspector

    alias_method :requires_grad?, :requires_grad

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

    def to_a
      reshape_arr(_flat_data, shape)
    end

    # TODO support dtype
    def to(device, non_blocking: false, copy: false)
      device = Device.new(device) if device.is_a?(String)
      _to(device, _dtype, non_blocking, copy)
    end

    def size(dim = nil)
      if dim
        _size_int(dim)
      else
        shape
      end
    end

    def shape
      dim.times.map { |i| size(i) }
    end

    # mirror Python len()
    def length
      size(0)
    end

    def item
      if numel != 1
        raise Error, "only one element tensors can be converted to Ruby scalars"
      end
      _flat_data.first
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
      cls.cast(_flat_data).reshape(*shape)
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

    def reshape(*size)
      # Python doesn't check if size == 1, just ignores later arguments
      size = size.first if size.size == 1 && size.first.is_a?(Array)
      _reshape(size)
    end

    def view(*size)
      size = size.first if size.size == 1 && size.first.is_a?(Array)
      _view(size)
    end

    # value and other are swapped for some methods
    def add!(value = 1, other)
      if other.is_a?(Numeric)
        _add__scalar(other, value)
      else
        _add__tensor(other, value)
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
      true_divide(other)
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
          result = result._select_int(dim, index)
        elsif index.is_a?(Range)
          finish = index.end
          finish += 1 unless index.exclude_end?
          result = result._slice_tensor(dim, index.begin, finish, 1)
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
        copy_to(_select_int(0, index), value)
      elsif index.is_a?(Range)
        finish = index.end
        finish += 1 unless index.exclude_end?
        copy_to(_slice_tensor(0, index.begin, finish, 1), value)
      else
        raise Error, "Unsupported index type: #{index.class.name}"
      end
    end

    def random!(from = 0, to)
      _random__from_to(from, to)
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
