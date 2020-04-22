require "mkmf-rice"

abort "Missing stdc++" unless have_library("stdc++")

$CXXFLAGS << " -std=c++14"

# change to 0 for Linux pre-cxx11 ABI version
$CXXFLAGS << " -D_GLIBCXX_USE_CXX11_ABI=1"

mac = RbConfig::CONFIG["host_os"] =~ /darwin/i

if have_library("omp") || have_library("gomp")
  $CXXFLAGS << " -DAT_PARALLEL_OPENMP=1"
  $CXXFLAGS << " -Xclang" if mac
  $CXXFLAGS << " -fopenmp"
end

# silence ruby/intern.h warning
$CXXFLAGS << " -Wno-deprecated-register"

# silence torch warnings
if mac
  $CXXFLAGS << " -Wno-shorten-64-to-32 -Wno-missing-noreturn"
else
  $CXXFLAGS << " -Wduplicated-cond"
end

inc, lib = dir_config("torch")

inc ||= "/usr/local/include"
lib ||= "/usr/local/lib"

$INCFLAGS << " -I#{inc}"
$INCFLAGS << " -I#{inc}/torch/csrc/api/include"

$LDFLAGS << " -Wl,-rpath,#{lib}"
$LDFLAGS << " -L#{lib}"

# https://github.com/pytorch/pytorch/blob/v1.5.0/torch/utils/cpp_extension.py#L1232-L1238
$LDFLAGS << " -lc10 -ltorch_cpu -ltorch"
$LDFLAGS << " -lc10_cuda -ltorch_cuda" if Dir["#{lib}/*torch_cuda*"].any?

# generate C++ functions
puts "Generating C++ functions..."
require_relative "../../lib/torch/native/generator"
Torch::Native::Generator.generate_cpp_functions

# create makefile
create_makefile("torch/ext")
