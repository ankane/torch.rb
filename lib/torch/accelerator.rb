module Torch
  module Accelerator
    class << self
      def current_accelerator(check_available: false)
        device = _current_device
        return nil unless device
        return nil if check_available && !available?
        device
      end

      def device_count
        _device_count
      end

      def available?
        _is_available
      end
    end
  end
end
