module Torch
  module Native
    class Function
      attr_reader :function, :tensor_options

      def initialize(function)
        @function = function

        # note: don't modify function in-place
        @tensor_options_str = ", *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None)"
        @tensor_options = @function["func"].include?(@tensor_options_str)
      end

      def func
        @func ||= @function["func"]
      end

      def name
        @name ||= func.split("(", 2).first
      end

      def python_module
        @python_module ||= @function["python_module"]
      end

      def variants
        @variants ||= (@function["variants"] || "function").split(", ")
      end

      def args
        @args ||= begin
          args = []
          pos = true
          args_str = func.sub(@tensor_options_str, ")").split("(", 2).last.split(") ->").first
          args_str.split(", ").each do |a|
            if a == "*"
              pos = false
              next
            end
            t, _, k = a.rpartition(" ")
            k, d = k.split("=")
            has_default = !d.nil?

            if d
              d =
                case d
                when "True"
                  true
                when "False"
                  false
                when "None"
                  nil
                when /\A\-?\d+\z/
                  d.to_i
                when "[]"
                  []
                when "[0,1]"
                  [0, 1]
                when /\A\de\-\d+\z/, /\A\d+\.\d+\z/
                  d.to_f
                when "Mean"
                  "mean"
                when "contiguous_format"
                  d
                when "long"
                  :long
                else
                  raise "Unknown default: #{d}"
                end
            end

            next if t == "Generator?"
            next if t == "MemoryFormat"
            next if t == "MemoryFormat?"
            args << {name: k.to_sym, type: t, default: d, pos: pos, has_default: has_default}
          end
          args
        end
      end

      def arg_checkers
        @arg_checkers ||= begin
          checkers = {}
          arg_types.each do |k, t|
            checker =
              case t
              when "Tensor"
                ->(v) { v.is_a?(Tensor) }
              when "Tensor?"
                ->(v) { v.nil? || v.is_a?(Tensor) }
              when "Tensor[]", "Tensor?[]"
                ->(v) { v.is_a?(Array) && v.all? { |v2| v2.is_a?(Tensor) } }
              when "int"
                if k == :reduction
                  ->(v) { v.is_a?(String) }
                else
                  ->(v) { v.is_a?(Integer) }
                end
              when "int?"
                ->(v) { v.is_a?(Integer) || v.nil? }
              when "float?"
                ->(v) { v.is_a?(Numeric) || v.nil? }
              when "bool?"
                ->(v) { v == true || v == false || v.nil? }
              when "float"
                ->(v) { v.is_a?(Numeric) }
              when /int\[.*\]/
                ->(v) { v.is_a?(Array) && v.all? { |v2| v2.is_a?(Integer) } }
              when "Scalar"
                ->(v) { v.is_a?(Numeric) }
              when "Scalar?"
                ->(v) { v.is_a?(Numeric) || v.nil? }
              when "ScalarType"
                ->(v) { false } # not supported yet
              when "ScalarType?"
                ->(v) { v.nil? }
              when "bool"
                ->(v) { v == true || v == false }
              when "str"
                ->(v) { v.is_a?(String) }
              else
                raise Error, "Unknown argument type: #{t}. Please report a bug with #{@name}."
              end
            checkers[k] = checker
          end
          checkers
        end
      end

      def int_array_lengths
        @int_array_lengths ||= begin
          ret = {}
          arg_types.each do |k, t|
            if t.match?(/\Aint\[.+\]\z/)
              size = t[4..-2]
              raise Error, "Unknown size: #{size}. Please report a bug with #{@name}." unless size =~ /\A\d+\z/
              ret[k] = size.to_i
            end
          end
          ret
        end
      end

      def arg_types
        @arg_types ||= args.map { |a| [a[:name], a[:type].split("(").first] }.to_h
      end

      def out_size
        @out_size ||= func.split("->").last.count("!")
      end

      def ret_size
        @ret_size ||= func.split("->").last.split(", ").size
      end

      def ret_array?
        @ret_array ||= func.split("->").last.include?('[]')
      end

      def out?
        out_size > 0 && base_name[-1] != "_"
      end

      def ruby_name
        @ruby_name ||= begin
          name = base_name
          if name.end_with?("_")
            "#{name[0..-2]}!"
          elsif name.start_with?("is_")
            "#{name[3..-1]}?"
          else
            name
          end
        end
      end

      def cpp_name
        @cpp_name ||= "_" + name.downcase.sub(".", "_")
      end

      def base_name
        @base_name ||= name.split(".").first
      end
    end
  end
end
