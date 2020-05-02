require_relative "test_helper"

class HubTest < Minitest::Test
  def test_list
    # TODO needs to be repo with Ruby models
    # entrypoints = Torch::Hub.list("pytorch/vision:v0.5.0")
  end

  def test_download_url_to_file
    dst = "#{Dir.mktmpdir}/test.html"
    assert_nil Torch::Hub.download_url_to_file("https://ankane.org/favicon.ico", dst)
    assert File.exist?(dst)
  end
end
