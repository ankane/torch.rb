require "mkmf-rice"

abort "Missing stdc++" unless have_library("stdc++")

$CXXFLAGS << " -std=c++11"

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

create_makefile("torch/ext")
