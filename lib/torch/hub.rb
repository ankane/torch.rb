module Torch
  module Hub
    class << self
      def list(github, force_reload: false)
        raise NotImplementedYet
      end

      def download_url_to_file(url, dst)
        require "open-uri"

        uri = URI.parse(url)
        raise "Invalid URL" unless uri.is_a?(URI::HTTP) # includes https

        puts "Downloading #{url}..."
        uri.open(max_redirects: 10) do |download|
          # TODO move file when possible
          IO.copy_stream(download, dst.to_str)
        end
        nil
      end

      def load_state_dict_from_url(url, model_dir: nil)
        unless model_dir
          torch_home = ENV["TORCH_HOME"] || "#{ENV["XDG_CACHE_HOME"] || "#{ENV["HOME"]}/.cache"}/torch"
          model_dir = File.join(torch_home, "checkpoints")
        end

        FileUtils.mkdir_p(model_dir)

        parts = URI(url)
        filename = File.basename(parts.path)
        cached_file = File.join(model_dir, filename)
        unless File.exist?(cached_file)
          # TODO support hash_prefix
          download_url_to_file(url, cached_file)
        end

        Torch.load(cached_file)
      end
    end
  end
end
