module Torch
  module Jit
    class << self
      alias _load load

      def load(path, device: nil)
        device_obj = device ? Torch.device(device) : nil

        _load(path, device_obj)
      end
    end
  end

  class ScriptModule
    alias call forward
  end
end
