require "mkmf-rice"

abort "Missing stdc++" unless have_library("stdc++")

$CXXFLAGS += " -std=c++14"

# change to 0 for Linux pre-cxx11 ABI version
$CXXFLAGS += " -D_GLIBCXX_USE_CXX11_ABI=1"

# TODO check compiler name
clang = RbConfig::CONFIG["host_os"] =~ /darwin/i

# check omp first
if have_library("omp") || have_library("gomp")
  $CXXFLAGS += " -DAT_PARALLEL_OPENMP=1"
  $CXXFLAGS += " -Xclang" if clang
  $CXXFLAGS += " -fopenmp"
end

if clang
  # silence ruby/intern.h warning
  $CXXFLAGS += " -Wno-deprecated-register"

  # silence torch warnings
  $CXXFLAGS += " -Wno-shorten-64-to-32 -Wno-missing-noreturn"
else
  # silence rice warnings
  $CXXFLAGS += " -Wno-noexcept-type"

  # silence torch warnings
  $CXXFLAGS += " -Wno-duplicated-cond -Wno-suggest-attribute=noreturn"
end

inc, lib = dir_config("torch")
inc ||= "/usr/local/include"
lib ||= "/usr/local/lib"

cuda_inc, cuda_lib = dir_config("cuda")
cuda_inc ||= "/usr/local/cuda/include"
cuda_lib ||= "/usr/local/cuda/lib64"

$LDFLAGS += " -L#{lib}" if Dir.exist?(lib)
abort "LibTorch not found" unless have_library("torch")

have_library("mkldnn")
have_library("nnpack")

with_cuda = false
if Dir["#{lib}/*torch_cuda*"].any?
  $LDFLAGS += " -L#{cuda_lib}" if Dir.exist?(cuda_lib)
  with_cuda = have_library("cuda") && have_library("cudnn")
end

$INCFLAGS += " -I#{inc}"
$INCFLAGS += " -I#{inc}/torch/csrc/api/include"

$LOAD_PATH.each do |x|
  if File.exist?(File.join(x, "numo/numo/narray.h"))
    $INCFLAGS += " -I#{x}/numo"
    break
  end
end
abort "Missing numo-narray" unless have_header("numo/narray.h")

$LDFLAGS += " -Wl,-rpath,#{lib}"
$LDFLAGS += ":#{cuda_lib}/stubs:#{cuda_lib}" if with_cuda

# https://github.com/pytorch/pytorch/blob/v1.5.0/torch/utils/cpp_extension.py#L1232-L1238
$LDFLAGS += " -lc10 -ltorch_cpu -ltorch"
if with_cuda
  $LDFLAGS += " -lcuda -lnvrtc -lnvToolsExt -lcudart -lc10_cuda -ltorch_cuda -lcufft -lcurand -lcublas -lcudnn"
  # TODO figure out why this is needed
  $LDFLAGS += " -Wl,--no-as-needed,#{lib}/libtorch.so"
end

# generate C++ functions
puts "Generating C++ functions..."
require_relative "../../lib/torch/native/generator"
Torch::Native::Generator.generate_cpp_functions

# create makefile
create_makefile("torch/ext")
