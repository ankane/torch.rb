# mirrors _tensor_str.py
module Torch
  module Inspector
    PRINT_OPTS = {
      precision: 4,
      threshold: 1000,
      edgeitems: 3,
      linewidth: 80,
      sci_mode: nil
    }

    class Formatter
      def initialize(tensor)
        @floating_dtype = tensor.floating_point?
        @complex_dtype = tensor.complex?
        @int_mode = true
        @sci_mode = false
        @max_width = 1

        tensor_view = Torch.no_grad { tensor.reshape(-1) }

        if !@floating_dtype
          tensor_view.each do |value|
            value_str = value.item.to_s
            @max_width = [@max_width, value_str.length].max
          end
        else
          nonzero_finite_vals = Torch.masked_select(tensor_view, Torch.isfinite(tensor_view) & tensor_view.ne(0))

          # no valid number, do nothing
          return if nonzero_finite_vals.numel == 0

          # Convert to double for easy calculation. HalfTensor overflows with 1e8, and there's no div() on CPU.
          nonzero_finite_abs = nonzero_finite_vals.abs.double
          nonzero_finite_min = nonzero_finite_abs.min.double
          nonzero_finite_max = nonzero_finite_abs.max.double

          nonzero_finite_vals.each do |value|
            if value.item != value.item.ceil
              @int_mode = false
              break
            end
          end

          if @int_mode
            # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
            # to indicate that the tensor is of floating type. add 1 to the len to account for this.
            if nonzero_finite_max / nonzero_finite_min > 1000.0 || nonzero_finite_max > 1.0e8
              @sci_mode = true
              nonzero_finite_vals.each do |value|
                value_str = "%.#{PRINT_OPTS[:precision]}e" % value.item
                @max_width = [@max_width, value_str.length].max
              end
            else
              nonzero_finite_vals.each do |value|
                value_str = "%.0f" % value.item
                @max_width = [@max_width, value_str.length + 1].max
              end
            end
          else
            # Check if scientific representation should be used.
            if nonzero_finite_max / nonzero_finite_min > 1000.0 || nonzero_finite_max > 1.0e8 || nonzero_finite_min < 1.0e-4
              @sci_mode = true
              nonzero_finite_vals.each do |value|
                value_str = "%.#{PRINT_OPTS[:precision]}e" % value.item
                @max_width = [@max_width, value_str.length].max
              end
            else
              nonzero_finite_vals.each do |value|
                value_str = "%.#{PRINT_OPTS[:precision]}f" % value.item
                @max_width = [@max_width, value_str.length].max
              end
            end
          end
        end

        @sci_mode = PRINT_OPTS[:sci_mode] unless PRINT_OPTS[:sci_mode].nil?
      end

      def width
        @max_width
      end

      def format(value)
        value = value.item

        if @floating_dtype
          if @sci_mode
            ret = "%#{@max_width}.#{PRINT_OPTS[:precision]}e" % value
          elsif @int_mode
            ret = String.new("%.0f" % value)
            unless value.infinite? || value.nan?
              ret += "."
            end
          else
            ret = "%.#{PRINT_OPTS[:precision]}f" % value
          end
        elsif @complex_dtype
          # TODO use float formatter for each part
          precision = PRINT_OPTS[:precision]
          imag = value.imag
          sign = imag >= 0 ? "+" : "-"
          ret = "%.#{precision}f#{sign}%.#{precision}fi" % [value.real, value.imag.abs]
        else
          ret = value.to_s
        end
        # Ruby throws error when negative, Python doesn't
        " " * [@max_width - ret.size, 0].max + ret
      end
    end

    def inspect
      Torch.no_grad do
        str_intern(self)
      end
    rescue => e
      # prevent stack error
      puts e.backtrace.join("\n")
      "Error inspecting tensor: #{e.inspect}"
    end

    private

    # TODO update
    def str_intern(slf)
      prefix = "tensor("
      indent = prefix.length
      suffixes = []

      has_default_dtype = [:float32, :int64, :bool].include?(slf.dtype)

      if slf.numel == 0 && !slf.sparse?
        # Explicitly print the shape if it is not (0,), to match NumPy behavior
        if slf.dim != 1
          suffixes << "size: #{shape.inspect}"
        end

        # In an empty tensor, there are no elements to infer if the dtype
        # should be int64, so it must be shown explicitly.
        if slf.dtype != :int64
          suffixes << "dtype: #{slf.dtype.inspect}"
        end
        tensor_str = "[]"
      else
        if !has_default_dtype
          suffixes << "dtype: #{slf.dtype.inspect}"
        end

        if slf.layout != :strided
          tensor_str = tensor_str(slf.to_dense, indent)
        else
          tensor_str = tensor_str(slf, indent)
        end
      end

      if slf.layout != :strided
        suffixes << "layout: #{slf.layout.inspect}"
      end

      # TODO show grad_fn
      if slf.requires_grad?
        suffixes << "requires_grad: true"
      end

      add_suffixes(prefix + tensor_str, suffixes, indent, slf.sparse?)
    end

    def add_suffixes(tensor_str, suffixes, indent, force_newline)
      tensor_strs = [tensor_str]
      # rfind in Python returns -1 when not found
      last_line_len = tensor_str.length - (tensor_str.rindex("\n") || -1) + 1
      suffixes.each do |suffix|
        suffix_len = suffix.length
        if force_newline || last_line_len + suffix_len + 2 > PRINT_OPTS[:linewidth]
          tensor_strs << ",\n" + " " * indent + suffix
          last_line_len = indent + suffix_len
          force_newline = false
        else
          tensor_strs.append(", " + suffix)
          last_line_len += suffix_len + 2
        end
      end
      tensor_strs.append(")")
      tensor_strs.join("")
    end

    def tensor_str(slf, indent)
      return "[]" if slf.numel == 0

      summarize = slf.numel > PRINT_OPTS[:threshold]

      if slf.dtype == :float16 || slf.dtype == :bfloat16
        slf = slf.float
      end
      formatter = Formatter.new(summarize ? summarized_data(slf) : slf)
      tensor_str_with_formatter(slf, indent, formatter, summarize)
    end

    def summarized_data(slf)
      edgeitems = PRINT_OPTS[:edgeitems]

      dim = slf.dim
      if dim == 0
        slf
      elsif dim == 1
        if size(0) > 2 * edgeitems
          Torch.cat([slf[0...edgeitems], slf[-edgeitems..-1]])
        else
          slf
        end
      elsif slf.size(0) > 2 * edgeitems
        start = edgeitems.times.map { |i| slf[i] }
        finish = (slf.length - edgeitems).upto(slf.length - 1).map { |i| slf[i] }
        Torch.stack((start + finish).map { |x| summarized_data(x) })
      else
        Torch.stack(slf.map { |x| summarized_data(x) })
      end
    end

    def tensor_str_with_formatter(slf, indent, formatter, summarize)
      edgeitems = PRINT_OPTS[:edgeitems]

      dim = slf.dim

      return scalar_str(slf, formatter) if dim == 0
      return vector_str(slf, indent, formatter, summarize) if dim == 1

      if summarize && slf.size(0) > 2 * edgeitems
        slices = (
          [edgeitems.times.map { |i| tensor_str_with_formatter(slf[i], indent + 1, formatter, summarize) }] +
          ["..."] +
          [((slf.length - edgeitems)...slf.length).map { |i| tensor_str_with_formatter(slf[i], indent + 1, formatter, summarize) }]
        )
      else
        slices = slf.size(0).times.map { |i| tensor_str_with_formatter(slf[i], indent + 1, formatter, summarize) }
      end

      tensor_str = slices.join("," + "\n" * (dim - 1) + " " * (indent + 1))
      "[" + tensor_str + "]"
    end

    def scalar_str(slf, formatter)
      formatter.format(slf)
    end

    def vector_str(slf, indent, formatter, summarize)
      # length includes spaces and comma between elements
      element_length = formatter.width + 2
      elements_per_line = [1, ((PRINT_OPTS[:linewidth] - indent) / element_length.to_f).floor.to_i].max
      char_per_line = element_length * elements_per_line

      if summarize && slf.size(0) > 2 * PRINT_OPTS[:edgeitems]
        data = (
          [slf[0...PRINT_OPTS[:edgeitems]].map { |val| formatter.format(val) }] +
          [" ..."] +
          [slf[-PRINT_OPTS[:edgeitems]..-1].map { |val| formatter.format(val) }]
        )
      else
        data = slf.map { |val| formatter.format(val) }
      end

      data_lines = (0...data.length).step(elements_per_line).map { |i| data[i...(i + elements_per_line)] }
      lines = data_lines.map { |line| line.join(", ") }
      "[" + lines.join("," + "\n" + " " * (indent + 1)) + "]"
    end
  end
end
