module Torch
  class Device
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
