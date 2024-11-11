module Torch
  class Device
    def index
      index? ? _index : nil
    end

    def inspect
      extra = ", index: #{index.inspect}" if index?
      "device(type: #{type.inspect}#{extra})"
    end
    alias_method :to_s, :inspect

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
end
