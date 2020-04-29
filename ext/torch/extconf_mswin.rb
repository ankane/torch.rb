$TORCHRB_PREFIX = File.join(File.dirname(File.expand_path(__FILE__)))

inc, lib = dir_config("torch")

inc ||= "C:/libtorch-win-shared-with-deps-1.5.0/libtorch/include"
lib ||= "C:/libtorch-win-shared-with-deps-1.5.0/libtorch/lib"

# generate C++ functions
puts "Generating C++ functions..."
require_relative "../../lib/torch/native/generator"
Torch::Native::Generator.generate_cpp_functions

target = 'ext'
source_dir = "../../../../ext/torch" if $0=~/\.\.\/\.\.\/\.\.\/\.\.\/ext\/torch/
run_install = source_dir ? "FALSE" : "TRUE"

cmakeliststxt = <<"CMAKELISTSTXT"
cmake_minimum_required(VERSION 3.12)
project(#{target})

message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
message(STATUS "CMAKE_BINARY_DIR: " ${CMAKE_BINARY_DIR})

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Zc:dllexportInlines- -EHsc")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

find_package(Torch REQUIRED)

file(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ext.cpp templates.cpp *_functions.cpp *_functions.hpp)
file(GLOB_RECURSE remove_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CMake* detect_cuda_*)
list(REMOVE_ITEM sources ${remove_sources})
message(STATUS "SOURCES: ${sources}")

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
target_link_libraries(#{target} PRIVATE c10)

#target_link_libraries(#{target} PRIVATE torch_cuda)
#target_link_libraries(#{target} PRIVATE c10_cuda)

if(#{run_install})
  install (TARGETS #{target} DESTINATION "#{$TORCHRB_PREFIX}/../../lib/torch")
endif()
CMAKELISTSTXT

fname = source_dir ? "#{source_dir}/CMakeLists.txt" : "CMakeLists.txt"
f = File.open(fname, mode="w")
f.write(cmakeliststxt)

#-----------------------------------------

source_dir = source_dir || "."

howtocompile = <<"HOWTOCOMPILE"
(Remove generated files and the CMakeFiles directory if necessary.)
cmake -S #{source_dir} -B . -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe" -DCMAKE_CXX_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe"
nmake
HOWTOCOMPILE

f = File.open("How_to_recompile_manually.txt", mode="w")
f.write(howtocompile)

#-----------------------------------------

system "cmake -S #{source_dir} -B . -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=\"c:/Program Files/LLVM/bin/clang-cl.exe\" -DCMAKE_CXX_COMPILER=\"c:/Program Files/LLVM/bin/clang-cl.exe\""

# Never go back to extconf.rb
exit
