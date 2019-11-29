# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
module Torch
  module Optim
    class Optimizer
      attr_reader :param_groups

      def initialize(params, defaults)
        @defaults = defaults
        @state = Hash.new { |hash, key| hash[key] = {} }
        @param_groups = []

        param_groups = params
        if param_groups.empty?
          raise ArgumentError, "optimizer got an empty parameter list"
        end
        if !param_groups[0].is_a?(Hash)
          param_groups = [{params: param_groups}]
        end

        param_groups.each do |param_group|
          add_param_group(param_group)
        end
      end

      def add_param_group(param_group)
        # TODO more advanced logic
        @param_groups << @defaults.merge(param_group)
      end

      def load_state_dict(state_dict)
        raise NotImplementedYet
      end

      def state_dict
        pack_group = lambda do |group|
          packed = group.select { |k, _| k != :params }.to_h
          packed[:params] = group[:params].map { |p| p.object_id }
          packed
        end

        param_groups = @param_groups.map { |g| pack_group.call(g) }
        packed_state = @state.map { |k, v| [k.is_a?(Tensor) ? k.object_id : k, v] }.to_h

        {
          state: packed_state,
          param_groups: param_groups
        }
      end

      def zero_grad
        @param_groups.each do |group|
          group[:params].each do |p|
            if p.grad
              p.grad.detach!
              p.grad.zero!
            end
          end
        end
      end
    end
  end
end
