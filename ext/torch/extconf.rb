require "mkmf-rice"

$CXXFLAGS += " -std=c++17 $(optflags)"

# change to 0 for Linux pre-cxx11 ABI version
$CXXFLAGS += " -D_GLIBCXX_USE_CXX11_ABI=1"

apple_clang = RbConfig::CONFIG["CC_VERSION_MESSAGE"] =~ /apple clang/i

if apple_clang
  # silence torch warnings
  $CXXFLAGS += " -Wno-deprecated-declarations"
else
  # silence rice warnings
  $CXXFLAGS += " -Wno-noexcept-type"

  # silence torch warnings
  $CXXFLAGS += " -Wno-duplicated-cond -Wno-suggest-attribute=noreturn"
end

paths = [
  "/usr/local",
  "/opt/homebrew",
  "/home/linuxbrew/.linuxbrew"
]

inc, lib = dir_config("torch")
inc ||= paths.map { |v| "#{v}/include" }.find { |v| Dir.exist?("#{v}/torch") }
lib ||= paths.map { |v| "#{v}/lib" }.find { |v| Dir["#{v}/*torch_cpu*"].any? }

unless inc && lib
  abort "LibTorch not found"
end

cuda_inc, cuda_lib = dir_config("cuda")
cuda_lib ||= "/usr/local/cuda/lib64"

cudnn_inc, cudnn_lib = dir_config("cudnn")
cudnn_lib ||= "/usr/local/cuda/lib"

$LDFLAGS += " -L#{lib}" if Dir.exist?(lib)
abort "LibTorch not found" unless have_library("torch")

have_library("mkldnn")
have_library("nnpack")

with_cuda = false
if Dir["#{lib}/*torch_cuda*"].any?
  $LDFLAGS += " -L#{cuda_lib}" if Dir.exist?(cuda_lib)
  $LDFLAGS += " -L#{cudnn_lib}" if Dir.exist?(cudnn_lib) && cudnn_lib != cuda_lib
  with_cuda = have_library("cuda") && have_library("cudnn")
end

$INCFLAGS += " -I#{inc}"
$INCFLAGS += " -I#{inc}/torch/csrc/api/include"

$LDFLAGS += " -Wl,-rpath,#{lib}"
if RbConfig::CONFIG["host_os"] =~ /darwin/i && RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i && Dir.exist?("/opt/homebrew/opt/libomp/lib")
  $LDFLAGS += ",-rpath,/opt/homebrew/opt/libomp/lib"
end
$LDFLAGS += ":#{cuda_lib}/stubs:#{cuda_lib}" if with_cuda

# https://github.com/pytorch/pytorch/blob/v2.9.0/torch/utils/cpp_extension.py#L1351-L1364
$LDFLAGS += " -lc10 -ltorch_cpu -ltorch"
if with_cuda
  $LDFLAGS += " -lcuda -lnvrtc"
  $LDFLAGS += " -lnvToolsExt" if File.exist?("#{cuda_lib}/libnvToolsExt.so")
  $LDFLAGS += " -lcudart -lc10_cuda -ltorch_cuda -lcufft -lcurand -lcublas -lcudnn"
  # TODO figure out why this is needed
  $LDFLAGS += " -Wl,--no-as-needed,#{lib}/libtorch.so"
end

# generate C++ functions
puts "Generating C++ functions..."
require_relative "../../codegen/generate_functions"
generate_functions

# create makefile
create_makefile("torch/ext")
