# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/adamax.py
module Torch
  module Optim
    class Adamax < Optimizer
      def initialize(params, lr: 2e-3, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid epsilon value: #{eps}" if eps < 0
        raise ArgumentError, "Invalid beta parameter at index 0: #{betas[0]}" if betas[0] < 0 || betas[0] >= 1
        raise ArgumentError, "Invalid beta parameter at index 1: #{betas[1]}" if betas[1] < 0 || betas[1] >= 1
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0

        defaults = {lr: lr, betas: betas, eps: eps, weight_decay: weight_decay}
        super(params, defaults)
      end

      def step(closure = nil)
        loss = nil
        if closure
          loss = closure.call
        end

        @param_groups.each do |group|
          group[:params].each do |p|
            next unless p.grad
            grad = p.grad.data
            if grad.sparse?
              raise Error, "Adamax does not support sparse gradients, please consider SparseAdam instead"
            end
            state = @state[p]

            # State initialization
            if state.size == 0
              state[:step] = 0
              state[:exp_avg] = Torch.zeros_like(p.data)
              state[:exp_inf] = Torch.zeros_like(p.data)
            end

            exp_avg, exp_inf = state[:exp_avg], state[:exp_inf]
            beta1, beta2 = group[:betas]
            eps = group[:eps]

            state[:step] += 1

            if group[:weight_decay] != 0
              grad = grad.add(group[:weight_decay], p.data)
            end

            # Update biased first moment estimate.
            exp_avg.mul!(beta1).add!(grad, alpha: 1 - beta1)
            # Update the exponentially weighted infinity norm.
            norm_buf = Torch.cat([
                exp_inf.mul!(beta2).unsqueeze(0),
                grad.abs.add!(eps).unsqueeze!(0)
            ], 0)
            Torch.max(norm_buf, 0, keepdim: false, out: [exp_inf, exp_inf.new.long])

            bias_correction = 1 - beta1 ** state[:step]
            clr = group[:lr] / bias_correction

            p.data.addcdiv!(-clr, exp_avg, exp_inf)
          end
        end

        loss
      end
    end
  end
end
