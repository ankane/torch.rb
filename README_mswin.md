# Torch.rb for Windows

If you want to use the LibTorch Windows library provided by PyTorch.org, 
you need to use the Ruby mswin version because PyTorch is developed in C++ and compiled with MSVC.

## Precondition

- Ruby mswin version
  - You will probably need to compile Ruby from source.
    The libraries required for Ruby mswin can be easily prepared with vcpkg.
    See https://github.com/Microsoft/vcpkg.git
- Visual Studio 2019 or lator
  - Use the toolset with Ruby and LibTorch compiled.
    CMake is necessary. So, select the option `C++ CMake tools for Windows` and install.
- LLVM clang-cl
  - I use LLVM-9.0.0-win64.exe which is from https://llvm.org/.
    Unfortunately I couldn't compile the Rice extension with MSVC.
- (Optional) CUDA Toolkit and cuDNN

## Installation

First, download the LibTorch file that matches your environment from PyTorch.org and place it anywhere.

```text
ex:
c:/libtorch-win-shared-with-deps-1.5.0/libtorch
|-- bin
|-- build-hash
|-- cmake
|-- include
|-- lib
|-- share
`-- test
```

Use a Command Prompt with MSVC, Ruby, LLVM, Git and LibTorch added to PATH.

```bat
ex:
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\libtorch-win-shared-with-deps-1.5.0\libtorch\bin;C:\libtorch-win-shared-with-deps-1.5.0\libtorch\lib;%PATH%
set PATH=C:\rubymswin-2.7.1-1\usr\bin;C:\home\vs2019\vcpkg\installed\x64-windows\bin;%PATH%
set PATH=C:\Program Files\LLVM\bin;%PATH%
set CC=c:/Program Files/LLVM/bin/clang-cl.exe
set CXX=c:/Program Files/LLVM/bin/clang-cl.exe
```

```bat
git clone https://github.com/golirev/torch.rb.git
cd torch.rb
git checkout gl_mswin_****  # Change to the branch you want
bundle install  # Requires Rice modified for Windows.
REM bundle exec rake install -- --with-torch-dir=/path/to/libtorch  # This option is not respected at least in mswin for now.
bundle exec gem install -- --with-torch-dir=/path/to/libtorch
```

Where `/path/to/libtorch` is `c:/libtorch-win-shared-with-deps-1.5.0/libtorch`.

It can take a few minutes to compile the extension.

