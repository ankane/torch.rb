class Function
  attr_reader :definition, :params, :retvals

  def initialize(definition)
    @definition = definition
    @params, @retvals = parse_func
  end

  def name
    func.split("(", 2).first
  end

  def base_name
    name.split(".").first
  end

  def func
    definition["func"]
  end

  def python_module
    definition["python_module"]
  end

  def variants
    (definition["variants"] || "function").split(", ")
  end

  def out_index
    params.index { |v| v[:modifier].to_s.include?("!") } if base_name[-1] != "_" && retvals.any?
  end

  def out?
    !out_index.nil?
  end

  private

  def parse_func
    input, output = func.split(/\s*->\s*/)
    [generate_params(input), generate_retvals(output)]
  end

  def generate_params(input)
    input = input.split("(", 2).last.chomp(")").split(/\s*,\s+/)

    keyword_only = false
    params = []
    input.each do |i|
      if i == "*"
        keyword_only = true
        next
      end

      type, name = i.split(/\s+/)

      if name.include?("=")
        name, default = name.split("=", 2)
      end

      optional = false
      if type.include?("?")
        optional = true
        type = type.delete("?")
      end

      type, modifier = extract_modifier(type)

      if type.include?("[")
        list_size = /\[(.*)\]/.match(type)[1]
        list_size = nil if list_size.empty?
      end

      params << {
        name: name,
        type: type,
        default: default,
        keyword_only: keyword_only,
        optional: optional,
        modifier: modifier,
        list_size: list_size
      }
    end

    if (params.map { |v| v[:name] } & ["dtype", "device", "layout", "pin_memory"]).size == 4
      params << {
        name: "requires_grad",
        type: "bool",
        default: "False",
        keyword_only: true,
        optional: true,
        modifier: nil,
        list_size: nil
      }
    end

    params
  end

  def generate_retvals(output)
    output =
      if output == "()"
        []
      elsif output[0] == "("
        output[1..-2].split(/\s*,\s*/)
      else
        [output]
      end

    retvals = []
    output.each do |o|
      type, name = o.split(/\s+/)
      type, modifier = extract_modifier(type)
      retvals << {name: name, type: type, modifier: modifier}
    end
    retvals
  end

  # Tensor(a), Tensor(a!), Tensor(a)[]
  def extract_modifier(type)
    if type.include?("(")
      parts = type.split(/[\(\)]/, 3)
      modifier = parts.delete_at(1)
      type = parts.join("")
    end
    [type, modifier]
  end
end
