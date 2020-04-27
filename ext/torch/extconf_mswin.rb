$TORCHRB_PREFIX = File.join(File.dirname(File.expand_path(__FILE__)))

inc, lib = dir_config("torch")

inc ||= "C:/libtorch-win-shared-with-deps-1.5.0/libtorch/include"
lib ||= "C:/libtorch-win-shared-with-deps-1.5.0/libtorch/lib"

# generate C++ functions
puts "Generating C++ functions..."
require_relative "../../lib/torch/native/generator"
Torch::Native::Generator.generate_cpp_functions

target = 'ext'

cmakeliststxt = <<"CMAKELISTSTXT"
cmake_minimum_required(VERSION 3.12)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Zc:dllexportInlines- -EHsc")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

project(#{target})
find_package(Torch REQUIRED)

file(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.hpp)
file(GLOB_RECURSE remove_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "build/*")
list(REMOVE_ITEM sources ${remove_sources})
message("${sources}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories("#{$RICE_PREFIX}/lib")
link_directories("#{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['libdir'])}")
link_directories("#{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['sitearchdir'])}")
link_directories("#{lib}")

add_library(#{target} SHARED ${sources})
set_target_properties(#{target} PROPERTIES OUTPUT_NAME "#{target}")
set_target_properties(#{target} PROPERTIES SUFFIX  ".so")

target_include_directories(#{target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(#{target} PRIVATE "#{$RICE_PREFIX}/include")
target_include_directories(#{target} PRIVATE "#{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['rubyhdrdir'])}")
target_include_directories(#{target} PRIVATE "#{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['rubyarchhdrdir'])}")
target_include_directories(#{target} PRIVATE "#{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['sitelibdir'])}")
target_include_directories(#{target} PRIVATE "#{inc}")
target_include_directories(#{target} PRIVATE "#{inc}/torch/csrc/api/include")

target_link_libraries(#{target} PRIVATE #{RbConfig.expand(RbConfig::MAKEFILE_CONFIG['RUBY_SO_NAME'])})
target_link_libraries(#{target} PUBLIC rice)
target_link_libraries(#{target} PRIVATE torch)
target_link_libraries(#{target} PRIVATE torch_cpu)
target_link_libraries(#{target} PRIVATE torch_cuda)
target_link_libraries(#{target} PRIVATE c10)
target_link_libraries(#{target} PRIVATE c10_cuda)

#target_link_libraries(#{target} PRIVATE asmjit)
#target_link_libraries(#{target} PRIVATE caffe2_detectron_ops_gpu)
#target_link_libraries(#{target} PRIVATE caffe2_module_test_dynamic)
#target_link_libraries(#{target} PRIVATE caffe2_nvrtc)
#target_link_libraries(#{target} PRIVATE clog)
#target_link_libraries(#{target} PRIVATE cpuinfo)
#target_link_libraries(#{target} PRIVATE fbgemm)
#target_link_libraries(#{target} PRIVATE libprotobuf)
#target_link_libraries(#{target} PRIVATE libprotobuf-lite)
#target_link_libraries(#{target} PRIVATE libprotoc)
#target_link_libraries(#{target} PRIVATE mkldnn)

install (TARGETS #{target} DESTINATION "#{$TORCHRB_PREFIX}/../../lib/torch")
CMAKELISTSTXT

f = File.open("CMakeLists.txt", mode="w")
f.write(cmakeliststxt)

#-----------------------------------------
howtocompile = <<"HOWTOCOMPILE"
(Remove the build directory)
mkdir build
cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe" -DCMAKE_CXX_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe"
nmake
HOWTOCOMPILE

f = File.open("How_to_recompile_manually.txt", mode="w")
f.write(howtocompile)

#-----------------------------------------

maketop = <<"MAKETOP"
all:
	pushd build & $(MAKE) /nologo /$(MAKEFLAGS) all & popd
	-copy /Y build\\ext.* .

clean:
	pushd build & $(MAKE) /nologo /$(MAKEFLAGS) clean & popd

clean-build:
	-rd /S /Q build
	-del /Q ..\\..\\lib\\torch\\ext.*

install:
	pushd build & $(MAKE) /nologo /$(MAKEFLAGS) install & popd

.PHONY: all clean clean-build install
MAKETOP

f = File.open("Makefile", mode="w")
f.write(maketop)


#-----------------------------------------
require 'fileutils'

build_dir = 'build'
FileUtils.mkdir(build_dir) unless File.directory?(build_dir)
FileUtils.cd(build_dir) do
  system "cmake .. -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=\"c:/Program Files/LLVM/bin/clang-cl.exe\" -DCMAKE_CXX_COMPILER=\"c:/Program Files/LLVM/bin/clang-cl.exe\""
end

# Never go back to extconf.rb
exit
