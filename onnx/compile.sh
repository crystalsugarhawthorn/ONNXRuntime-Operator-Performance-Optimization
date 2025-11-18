#!/bin/bash

# 编译HIP kernel文件
/opt/dtk/hip/bin/hipcc -w --offload-arch=gfx928 \
    -I/opt/dtk-25.04/include \
    -I/opt/dtk/hip/include \
    -I/opt/dtk/hipblas/include \
    -I/opt/dtk/rocblas/include \
    -I/opt/dtk/miopen/include \
    -fPIC -x hip \
    -o rocm_ops.hip.o -c rocm_ops.hip.cpp

/opt/dtk/hip/bin/hipcc -w --offload-arch=gfx928 \
    -I/opt/dtk-25.04/include \
    -I/opt/dtk/hip/include \
    -I/opt/dtk/hipblas/include \
    -I/opt/dtk/rocblas/include \
    -fPIC -x hip \
    -o winograd_conv2d.hip.o -c winograd/winograd_conv2d.hip.cpp

/opt/dtk/hip/bin/hipcc -w --offload-arch=gfx928 \
    -I/opt/dtk-25.04/include \
    -I/opt/dtk/hip/include \
    -I/opt/dtk/hipblas/include \
    -I/opt/dtk/rocblas/include \
    -fPIC -x hip \
    -o winograd_f2x3.hip.o -c winograd/winograd_f2x3.hip.cpp

/opt/dtk/hip/bin/hipcc -w --offload-arch=gfx928 \
    -I/opt/dtk-25.04/include \
    -I/opt/dtk/hip/include \
    -I/opt/dtk/hipblas/include \
    -I/opt/dtk/rocblas/include \
    -fPIC -x hip \
    -o winograd_f4x3.hip.o -c winograd/winograd_f4x3.hip.cpp

# 编译C++文件
/usr/bin/c++ -w -DUSE_ROCM=1 \
    -I ./include/onnxruntime/ \
    -fPIC "-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1" \
    -o rocm_ops.cc.o -c rocm_ops.cc

/usr/bin/c++ -w -I./include/onnxruntime/ \
    -fPIC \
    -o custom_op_library.cc.o -c custom_op_library.cc

# 链接生成动态库（添加hipBLAS库）
/opt/dtk/llvm/bin/clang++ -w -fPIC -shared \
    -Wl,-soname,libcustom_op_library.so \
    -o libcustom_op_library.so \
    rocm_ops.hip.o winograd_conv2d.hip.o winograd_f2x3.hip.o winograd_f4x3.hip.o \
    custom_op_library.cc.o rocm_ops.cc.o \
    -L/opt/dtk/lib \
    -L/opt/dtk/hipblas/lib \
    -L/opt/dtk/rocblas/lib \
    -L/opt/dtk/miopen/lib \
    -Wl,-rpath,/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/hipblas/lib:/opt/dtk/rocblas/lib \
    /opt/dtk/hip/lib/libgalaxyhip.so.5.2.25085.1211-205b0686 \
    -lhipblas \
    -lrocblas \
    -lMIOpen  \
    /opt/dtk/llvm/lib/clang/15.0.0/lib/linux/libclang_rt.builtins-x86_64.a \
    -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc