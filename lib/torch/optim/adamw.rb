# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py
module Torch
  module Optim
    class AdamW < Optimizer
      def initialize(params, lr: 1e-3, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 1e-2, amsgrad: false)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid epsilon value: #{eps}" if eps < 0
        raise ArgumentError, "Invalid beta parameter at index 0: #{betas[0]}" if betas[0] < 0 || betas[0] >= 1
        raise ArgumentError, "Invalid beta parameter at index 1: #{betas[1]}" if betas[1] < 0 || betas[1] >= 1

        defaults = {lr: lr, betas: betas, eps: eps, weight_decay: weight_decay, amsgrad: amsgrad}
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

            # Perform stepweight decay
            p.data.mul!(1 - group[:lr] * group[:weight_decay])

            # Perform optimization step
            grad = p.grad.data
            if grad.sparse?
              raise Error, "AdamW does not support sparse gradients, please consider SparseAdam instead"
            end
            amsgrad = group[:amsgrad]

            state = @state[p]

            # State initialization
            if state.size == 0
              state[:step] = 0
              # Exponential moving average of gradient values
              state[:exp_avg] = Torch.zeros_like(p.data)
              # Exponential moving average of squared gradient values
              state[:exp_avg_sq] = Torch.zeros_like(p.data)
              if amsgrad
                # Maintains max of all exp. moving avg. of sq. grad. values
                state[:max_exp_avg_sq] = Torch.zeros_like(p.data)
              end
            end

            exp_avg, exp_avg_sq = state[:exp_avg], state[:exp_avg_sq]
            if amsgrad
              max_exp_avg_sq = state[:max_exp_avg_sq]
            end
            beta1, beta2 = group[:betas]

            state[:step] += 1
            bias_correction1 = 1 - beta1 ** state[:step]
            bias_correction2 = 1 - beta2 ** state[:step]

            # Decay the first and second moment running average coefficient
            exp_avg.mul!(beta1).add!(grad, alpha: 1 - beta1)
            exp_avg_sq.mul!(beta2).addcmul!(1 - beta2, grad, grad)
            if amsgrad
              # Maintains the maximum of all 2nd moment running avg. till now
              Torch.max(max_exp_avg_sq, exp_avg_sq, out: max_exp_avg_sq)
              # Use the max. for normalizing running avg. of gradient
              denom = (max_exp_avg_sq.sqrt / Math.sqrt(bias_correction2)).add!(group[:eps])
            else
              denom = (exp_avg_sq.sqrt / Math.sqrt(bias_correction2)).add!(group[:eps])
            end

            step_size = group[:lr] / bias_correction1

            p.data.addcdiv!(-step_size, exp_avg, denom)
          end
        end

        loss
      end
    end
  end
end
