# We use a generic interface for methods (*args, **options)
# and this class to determine the C++ method to call
#
# This is needed since LibTorch uses function overloading,
# which isn't available in Ruby or Python
#
# PyTorch uses this approach, but the parser/dispatcher is written in C++
#
# We could generate Ruby methods directly, but an advantage of this approach is
# arguments and keyword arguments can be used interchangably like in Python,
# making it easier to port code

module Torch
  module Native
    module Dispatcher
      class << self
        def bind
          functions = Generator.grouped_functions
          bind_functions(::Torch, :define_singleton_method, functions[:torch])
          bind_functions(::Torch::Tensor, :define_method, functions[:tensor])
          bind_functions(::Torch::NN, :define_singleton_method, functions[:nn])
        end

        def bind_functions(context, def_method, functions)
          functions.group_by(&:ruby_name).sort_by { |g, _| g }.each do |name, funcs|
            if def_method == :define_method
              funcs.map! { |f| Function.new(f.function) }
              funcs.each { |f| f.args.reject! { |a| a[:name] == :self } }
            end

            defined = def_method == :define_method ? context.method_defined?(name) : context.respond_to?(name)
            next if defined && name != "clone"

            parser = Parser.new(funcs)

            context.send(def_method, name) do |*args, **options|
              result = parser.parse(args, options)
              raise ArgumentError, result[:error] if result[:error]
              send(result[:name], *result[:args])
            end
          end
        end
      end
    end
  end
end

Torch::Native::Dispatcher.bind
