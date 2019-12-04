module Torch
  module Native
    class Function
      def initialize(function)
        @function = function
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

      def args_str
        @args_str ||= func.split("(", 2).last.split(") ->").first
      end

      # TODO clean up
      def args
        @args ||= begin
          args = args_str.split(", ").map { |a| a.split(" ").last }.map { |a| a.split("=").first }
          args.delete("*")
          args
        end
      end

      def out_size
        @out_size ||= func.split("->").last.count("!")
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

      def match(args, options)
        fargs = args().dup

        # check options first
        values = {}
        unknown_options = []
        options.each do |k, v|
          if fargs.include?(k)
            fargs.delete(k)
            values[k.to_s] = v
          else
            unknown_options << k
          end
        end

        if unknown_options.any?
          return nil # unknown arguments
        end

        # TODO separate out positional arguments
        if args.size > fargs.size
          return nil # too many arguments
        end

        args.each_with_index do |a, i|
          values[fargs[i]] = a
        end
        fargs = fargs[args.size..-1]

        # TODO check if any required fargs remains
        required_args = []
        missing_args = fargs.select { |a| required_args.include?(a) }
        if missing_args.any?
          return nil # missing args
        end

        # TODO set default values
        default_values = {}
        default_values.each do |k, v|
          values[k] ||= v
        end

        # we have a match
        [cpp_name] + self.args.map { |a| values[a] }
      end
    end
  end
end
