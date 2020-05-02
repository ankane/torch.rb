module Torch
  module Hub
    class << self
      def list(github, force_reload: false)
        raise NotImplementedYet
      end

      # TODO handle redirects
      def download_url_to_file(url, dst)
        uri = URI(url)
        tmp = "#{Dir.tmpdir}/#{Time.now.to_f}" # TODO better name

        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
          request = Net::HTTP::Get.new(uri)

          puts "Downloading #{url}..."
          File.open(tmp, "wb") do |f|
            http.request(request) do |response|
              response.read_body do |chunk|
                f.write(chunk)
              end
            end
          end
        end

        FileUtils.mv(tmp, dst)

        nil
      end

      def load_state_dict_from_url(url)
        raise NotImplementedYet
      end
    end
  end
end
