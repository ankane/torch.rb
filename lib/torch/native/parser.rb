module Torch
  module Native
    class Parser
      def initialize(functions)
        @functions = functions
        @name = @functions.first.ruby_name
        @min_args = @functions.map { |f| f.args.count { |a| a[:pos] && !a[:has_default] } }.min
        @max_args = @functions.map { |f| f.args.count { |a| a[:pos] } }.max
        @int_array_first = @functions.all? { |c| c.args.first && c.args.first[:type] == "int[]" }
      end

      def parse(args, options)
        candidates = @functions.dup

        # TODO check candidates individually to see if they match
        if @int_array_first
          int_args = []
          while args.first.is_a?(Integer)
            int_args << args.shift
          end
          if int_args.any?
            raise ArgumentError, "argument '#{candidates.first.args.first[:name]}' must be array of ints, but found element of type #{args.first.class.name} at pos #{int_args.size + 1}" if args.any?
            args.unshift(int_args)
          end
        end

        # TODO account for args passed as options here
        if args.size < @min_args || args.size > @max_args
          expected = String.new(@min_args.to_s)
          expected += "..#{@max_args}" if @max_args != @min_args
          return {error: "wrong number of arguments (given #{args.size}, expected #{expected})"}
        end

        candidates.reject! { |f| args.size > f.args.size }

        # handle out with multiple
        # there should only be one match, so safe to modify all
        if options[:out]
          if (out_func = candidates.find { |f| f.out? }) && out_func.out_size > 1
            out_args = out_func.args.last(2).map { |a| a[:name] }
            out_args.zip(options.delete(:out)).each do |k, v|
              options[k.to_sym] = v
            end
            candidates = [out_func]
          end
        else
          # exclude functions missing required options
          candidates.reject!(&:out?)
        end

        # exclude functions where options don't match
        options.each do |k, v|
          candidates.select! do |func|
            func.args.any? { |a| a[:name] == k.to_s }
          end
          # TODO show all bad keywords at once like Ruby?
          return {error: "unknown keyword: #{k}"} if candidates.empty?
        end

        final_values = nil

        # check args
        candidates.select! do |func|
          good = true

          # set values
          # TODO use array instead of hash?
          values = {}
          args.each_with_index do |a, i|
            values[func.args[i][:name]] = a
          end
          options.each do |k, v|
            # TODO use symbols to avoid allocation
            values[k.to_s] = v
          end
          func.args.each do |fa|
            values[fa[:name]] = fa[:default] if values[fa[:name]].nil?
          end
          func.int_array_lengths.each do |k, len|
            values[k] = [values[k]] * len if values[k].is_a?(Integer)
          end

          arg_checkers = func.arg_checkers

          values.each_key do |k|
            good = arg_checkers[k].call(values[k])
            if !good
              if candidates.size == 1
                t = func.arg_types[k]
                k = "input" if k == "self"
                return {error: "#{@name}(): argument '#{k}' must be #{t}"}
              end
              break
            end
          end

          if good
            final_values = values
          end

          good
        end

        if candidates.size != 1
          raise Error, "This should never happen. Please report a bug with #{@name}."
        end

        func = candidates.first
        args = func.args.map { |a| final_values[a[:name]] }
        args << TensorOptions.new.dtype(6) if func.tensor_options
        {
          name: func.cpp_name,
          args: args
        }
      end
    end
  end
end
