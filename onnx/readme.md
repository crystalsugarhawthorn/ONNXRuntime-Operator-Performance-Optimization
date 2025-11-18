# Custom Op ONNXRuntime
## Purpose
`Adding the custom operator implementation and registering it in ONNX Runtime`

## 环境配置
见文档《OnnxRuntime赛题的环境配置说明》

## 使用
### 编译工程
```
cd ONNXRuntime
bash compile.sh     // 编译生成算子库
```

### 运行示例
1. 目录
```
ONNXRuntime/
├── model/                         # 测试模型目录（不可更改）
│
└── onnx/                          # 主工程目录
    ├── baseline_latency.json      # 各模型 baseline （不可更改）
    ├── benchmark.py               # 性能测试脚本（不可更改）
    ├── compile.sh                 # 编译脚本
    ├── custom_op_library.cc       # 注册自定义算子
    ├── custom_op_library.h
    ├── cuda_utils.py              
    ├── fp16.py                    
    ├── include/                   # ONNX Runtime 头文件
    ├── node_utils.py              
    ├── readme.md                  # 项目说明
    ├── rocm_ops.cc                # ROCm 调度逻辑
    ├── rocm_ops.h
    ├── rocm_ops.hip.cpp           # 自定义算子实现
    ├── libcustom_op_library.so    # 编译生成的自定义算子库
    └── validatone.sh              # 验证精度和性能脚本
```

2. 执行步骤
```
bash compile.sh          # 编译自定义算子与项目代码  
bash validatione.sh      # 验证算子功能的正确性与性能表现
```

3. 备注
```
1.在compile.sh中，选项--offload-arch=gfx906，请将gfx906替换为本机适配的rocm架构(可用rocminfo | grep gfx查看)。
2.若需测试别的自定义算子，修改内容如下：
    2.1 修改rocm_ops.hip，重新实现自定义算子
    2.2 修改rocm_ops.cc, 修改对自定义算子的调用
3.在提交至评分系统之前，请务必在本地完成代码的编译和算子功能正确性验证，确保各项测试通过后再进行提交。仅需提交onnx目录即可，model目录请勿上传。
```

## 参考资料
- https://github.com/microsoft/onnxruntime
- https://onnxruntime.ai/docs/extensions/add-op.html
