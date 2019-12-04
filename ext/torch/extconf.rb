require "mkmf-rice"

abort "Missing stdc++" unless have_library("stdc++")

$CXXFLAGS << " -std=c++11"

# needed for Linux pre-cxx11 ABI version
# $CXXFLAGS << " -D_GLIBCXX_USE_CXX11_ABI=0"

# silence ruby/intern.h warning
$CXXFLAGS << " -Wno-deprecated-register"

inc, lib = dir_config("torch")

inc ||= "/usr/local/include"
lib ||= "/usr/local/lib"

$INCFLAGS << " -I#{inc}"
$INCFLAGS << " -I#{inc}/torch/csrc/api/include"

$LDFLAGS << " -Wl,-rpath,#{lib}"
$LDFLAGS << " -L#{lib}"
$LDFLAGS << " -ltorch -lc10"

# generate C++ functions
require_relative "../../lib/torch/native/dispatcher"
Torch::Native::Dispatcher.generate_cpp_functions

# create makefile
create_makefile("torch/ext")
