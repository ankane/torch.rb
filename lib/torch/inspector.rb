module Torch
  module Inspector
    # TODO make more performant, especially when summarizing
    # how? only read data that will be displayed
    def inspect
      data =
        if numel == 0
          "[]"
        elsif dim == 0
          item
        else
          summarize = numel > 1000

          if dtype == :bool
            fmt = "%s"
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
              sci = max > 1e8 || max < 1e-4

              all_int = values.all? { |v| v.finite? && v == v.to_i }
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
          end

          inspect_level(to_a, fmt, dim - 1, 0, summarize)
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

    # TODO DRY code
    def inspect_level(arr, fmt, total, level, summarize)
      if level == total
        cols =
          if summarize && arr.size > 7
            arr[0..2].map { |v| fmt % v } +
            ["..."] +
            arr[-3..-1].map { |v| fmt % v }
          else
            arr.map { |v| fmt % v }
          end

        "[#{cols.join(", ")}]"
      else
        rows =
          if summarize && arr.size > 7
            arr[0..2].map { |row| inspect_level(row, fmt, total, level + 1, summarize) } +
            ["..."] +
            arr[-3..-1].map { |row| inspect_level(row, fmt, total, level + 1, summarize) }
          else
            arr.map { |row| inspect_level(row, fmt, total, level + 1, summarize) }
          end

        "[#{rows.join(",#{"\n" * (total - level)}#{" " * (level + 8)}")}]"
      end
    end
  end
end
