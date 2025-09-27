clean and rebuild:
rm -rf build
cmake -S . -B build
cmake --build build --parallel

build version:
rm -rf build-debug
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel

release
rm -rf build-release
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release cmake --build build-release --parallel

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release  -DCMAKE_CXX_FLAGS_RELEASE="-O3,-DNDEBUG"
cmake --build build-release --parallel

test:
ctest --test-dir build --output-on-failure

Installation vcpkg:
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh        # macOS/Linux
.\bootstrap-vcpkg.bat       # Windows PowerShell

Installing Eigen3:
./vcpkg install eigen3

Headers will be installed in:
vcpkg/installed/<triplet>/include/eigen3/Eigen/...

Tell CMake to use the vcpkg toolchain file when configuring:
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake