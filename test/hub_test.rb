require_relative "test_helper"

class HubTest < Minitest::Test
  def test_list
    # TODO needs to be repo with Ruby models
    # entrypoints = Torch::Hub.list("pytorch/vision:v0.5.0")
  end

  def test_download_url_to_file
    dst = File.join(Dir.tmpdir, "tensor.pth")
    assert_nil Torch::Hub.download_url_to_file(url, dst)
    assert File.exist?(dst)
  end

  def test_load_state_dict_from_url
    state_dict = Torch::Hub.load_state_dict_from_url(url)
    assert_equal Torch.load("test/support/tensor.pth").to_a, state_dict.to_a
  end

  def url
    "https://github.com/ankane/torch.rb/raw/master/test/support/tensor.pth"
  end
end
