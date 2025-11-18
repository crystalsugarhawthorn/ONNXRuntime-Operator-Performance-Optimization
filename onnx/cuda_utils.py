import onnxruntime as ort
import onnx
import numpy as np
import time
from node_utils import node_utils, INPUT_TYPE_TO_NP_TYPE_MAP
import sys

def set_batchsize(model, batchSize):
    for node in model.graph.node:
        if node.op_type in ['Reshape', 'Split', 'Transpose']:
            return model

    del model.graph.value_info[:]

    for input in model.graph.input:
        if len(input.type.tensor_type.shape.dim) > 1:
            input.type.tensor_type.shape.dim[0].dim_value = batchSize

    for output in model.graph.output:
        if len(output.type.tensor_type.shape.dim) > 1:
            output.type.tensor_type.shape.dim[0].dim_value = batchSize

    return model

def model_setbs(model, batchSize):
    del model.graph.value_info[:]

    for input in model.graph.input:
        if len(input.type.tensor_type.shape.dim) > 1:
            input.type.tensor_type.shape.dim[0].dim_value = batchSize

    for output in model.graph.output:
        if len(output.type.tensor_type.shape.dim) > 1:
            output.type.tensor_type.shape.dim[0].dim_value = batchSize

    return model

def model_run(modelPath, batchSize=None):
    model = onnx.load(modelPath)
    if batchSize is not None:
        model = set_batchsize(model, batchSize)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    EP_list = ['CUDAExecutionProvider']
    start = time.time()
    cuda_session = ort.InferenceSession(model.SerializeToString(), providers=EP_list, sess_options=session_options)
    end = time.time()
    duration = (end - start) * 1000
    print("Initialize Session cost {} ms".format(duration))

    inputs = cuda_session.get_inputs()
    outputs = cuda_session.get_outputs()

    input_dict = {}
    for input in inputs:
        shape = [s for s in input.shape]
        for idx in range(len(shape)):
            if shape[idx] is None:
                print("[ERROR] Input shape invalid,please Check")
                return -1
        input_data = np.random.random(shape)
        if input.type.find('int') > 0:
            input_data = input_data*10
        input_data = input_data.astype(INPUT_TYPE_TO_NP_TYPE_MAP[input.type])
        input_dict[input.name] = ort.OrtValue.ortvalue_from_numpy(input_data, 'cuda_pinned', 0)

    outputs_names = []
    for output in outputs:
        outputs_names.append(output.name)

    io_binding = cuda_session.io_binding()
    for key, ortValue in input_dict.items():
        io_binding.bind_ortvalue_input(key, ortValue)

    for out_name in outputs_names:
        io_binding.bind_output(out_name, 'cuda_pinned', device_id=0)

    # warm up
    warm_up_num = 20
    start = time.time()
    for i in range(warm_up_num):
        cuda_session.run_with_iobinding(io_binding)
    end = time.time()
    duration = (end - start) / (batchSize * warm_up_num) * 1000
    print("Warm up cost {} ms".format(duration))

    run_num = 50
    start = time.time()
    for i in range(run_num):
        cuda_session.run_with_iobinding(io_binding)
    end = time.time()
    duration = (end - start) * 1000 / (run_num * batchSize)
    print("Current inference cost {} ms".format(duration))
    print("FPS is {:.2f}".format(1000/duration))
    del cuda_session
    return duration

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print("Input parameter error...")
        print("python cudaRun.py modelPath batchSize")
        sys.exit()

    modelPath = sys.argv[1]
    batchSize = int(sys.argv[2])
    print(modelPath)
    model_run(modelPath, batchSize)
