module Torch
  class Device
    def index
      index? ? _index : nil
    end

    def inspect
      extra = ", index: #{index.inspect}" if index?
      "device(type: #{type.inspect}#{extra})"
    end

    def to_s
      _str
    end

    def ==(other)
      eql?(other)
    end

    def eql?(other)
      other.is_a?(Device) && other.type == type && other.index == index
    end

    def hash
      [type, index].hash
    end
  end

  # String-like wrapper that also exposes device metadata
  class DeviceString < String
    def initialize(device)
      @device = device
      super(device._str)
    end

    def type
      @device.type
    end

    def index
      @device.index
    end
  end
end
