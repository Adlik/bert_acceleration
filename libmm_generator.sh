export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
cd ./FBGEMM
g++  ./bench/BenchUtils.cc ../accelerate_mm/accelerate_mm.cc -DCPUINFO_SUPPORTED_PLATFORM=1 -I/usr/local/include/fbgemm/ -I./ -I./include/fbgemm -L/usr/local/lib  -lfbgemm -lasmjit -lcpuinfo -lclog -lgmock -lgmock_main -fopenmp -O3 -DNDEBUG -m64 -mavx2 -mfma -masm=intel -DUSE_BLAS -DASMJIT_STATIC -std=c++11  -shared -fPIC -o ../accelerate_mm/libmm.so
echo "Successfully generate libmm.so to path \"./accelerate_mm/libmm.so\""