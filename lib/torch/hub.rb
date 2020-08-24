module Torch
  module Hub
    class << self
      def list(github, force_reload: false)
        raise NotImplementedYet
      end

      def download_url_to_file(url, dst)
        uri = URI(url)
        tmp = nil
        location = nil

        puts "Downloading #{url}..."
        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
          request = Net::HTTP::Get.new(uri)

          http.request(request) do |response|
            case response
            when Net::HTTPRedirection
              location = response["location"]
            when Net::HTTPSuccess
              tmp = "#{Dir.tmpdir}/#{Time.now.to_f}" # TODO better name
              File.open(tmp, "wb") do |f|
                response.read_body do |chunk|
                  f.write(chunk)
                end
              end
            else
              raise Error, "Bad response"
            end
          end
        end

        if location
          download_url_to_file(location, dst)
        else
          FileUtils.mv(tmp, dst)
          nil
        end
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
