module Torch
  module Inspector
    def inspect
      data =
        if numel == 0
          "[]"
        elsif dim == 0
          to_a.first
        else
          values = to_a.flatten
          abs = values.select { |v| v != 0 }.map(&:abs)
          max = abs.max || 1
          min = abs.min || 1

          total = 0
          if values.any? { |v| v < 0 }
            total += 1
          end

          if floating_point?
            sci = max / min.to_f > 1000 || max > 1e8 || min < 1e-4

            all_int = values.all? { |v| v == v.to_i }
            decimal = all_int ? 1 : 4

            total += sci ? 10 : decimal + 1 + max.to_i.to_s.size

            if sci
              fmt = "%#{total}.4e"
            else
              fmt = "%#{total}.#{decimal}f"
            end
          else
            total += max.to_s.size
            fmt = "%#{total}d"
          end

          inspect_level(to_a, fmt, dim - 1)
        end

      attributes = []
      if requires_grad
        attributes << "requires_grad: true"
      end
      if ![:float32, :int64, :bool].include?(dtype)
        attributes << "dtype: #{dtype.inspect}"
      end

      "tensor(#{data}#{attributes.map { |a| ", #{a}" }.join("")})"
    end

    private

    def inspect_level(arr, fmt, total, level = 0)
      if level == total
        "[#{arr.map { |v| fmt % v }.join(", ")}]"
      else
        "[#{arr.map { |row| inspect_level(row, fmt, total, level + 1) }.join(",#{"\n" * (total - level)}#{" " * (level + 8)}")}]"
      end
    end
  end
end
