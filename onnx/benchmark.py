import onnxruntime as ort
import numpy as np
import os
import sys
import onnx
from onnx import numpy_helper
from node_utils import INPUT_TYPE_TO_NP_TYPE_MAP
from cuda_utils import set_batchsize, model_setbs
from scipy import spatial
import argparse
import time

def save_input_output_data(save_path, data_dict, isInput=True):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    keys = list(data_dict.keys())
    data_prefix = 'input'
    if not isInput:
        data_prefix = 'output'

    for j in range(len(data_dict)):
        with open(os.path.join(save_path, '{}_{}.pb'.format(data_prefix, j)), 'wb') as f:
            f.write(numpy_helper.from_array(
                data_dict[keys[j]], keys[j]).SerializeToString())

def load_pb_data(pb_path):
    with open(pb_path, 'rb') as f:
        input_content = f.read()
        tensor = onnx.TensorProto()
        tensor.ParseFromString(input_content)
        f.close()
        return numpy_helper.to_array(tensor)

def get_cosine(gpu_array, cpu_array):
    gpu_array = gpu_array.astype(np.float64)
    cpu_array = cpu_array.astype(np.float64)
    gpu_array = gpu_array.reshape([-1])
    cpu_array = cpu_array.reshape([-1])
    cosine = spatial.distance.cosine(cpu_array, gpu_array)
    return cosine

def get_snr(gpu_array, cpu_array):
    cpu_array = cpu_array.astype(np.float64)
    gpu_array = gpu_array.astype(np.float64)
    diff_array = cpu_array - gpu_array
    x = diff_array * diff_array
    x = np.sum(x)
    y = cpu_array * cpu_array
    y = np.sum(y)
    snr = (x) / (y + 1e-7)
    snr = np.mean(snr)
    return snr

def accuracy_check_run(args):
    EP_List = ['ROCMExecutionProvider']
    model = onnx.load(args.input_model)
    model = set_batchsize(model, args.batchsize)
    so = ort.SessionOptions()
    so.enable_profiling = True

    lib_path = os.path.join(os.getcwd(), "libcustom_op_library.so")
    so.register_custom_ops_library(lib_path)

    
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    cuda_session = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=EP_List
    )
    #cuda_session = ort.InferenceSession(model.SerializeToString(), providers=EP_List)
    inputs = cuda_session.get_inputs()
    outputs = cuda_session.get_outputs()

    file_list = os.listdir(args.datapath)
    input_list = []
    output_list = []
    for file in file_list:
        if file[:5] == 'input':
            input_list.append(file)
        elif file[:6] == 'output':
            output_list.append(file)

    input_traits = [int(i[6:-3]) for i in input_list]
    input_traits = sorted(input_traits)
    input_list = [os.path.join(args.datapath, "input_{}.pb".format(i)) for i in input_traits]
    output_traits = [int(i[7:-3]) for i in output_list]
    output_traits = sorted(output_traits)
    output_list = [os.path.join(args.datapath, "output_{}.pb".format(i)) for i in output_traits]

    input_dict = {}
    for input, input_file in zip(inputs, input_list):
        input_dict[input.name] = load_pb_data(input_file)
        if input_dict[input.name].shape[0] != args.batchsize:
            print("Batchsize error! input data batchsize is {} but your input batchsize is {}, Please fix!".format(input_dict[input_file[0].name].shape[0], args.batchsize))
            sys.exit()

    gt_dict = {}
    for output, gt_file in zip(outputs, output_list):
        gt_dict[output.name] = load_pb_data(gt_file)

    output_names = [x.name for x in cuda_session.get_outputs()]
    output_data = cuda_session.run(output_names, input_dict)

    for idx, output_name in enumerate(output_names):
        print("output {}".format(output_name))
        print("SNR IS : {}".format(get_snr(gt_dict[output_name], output_data[idx])))
        print("COSINE IS : {}\n".format(get_cosine(gt_dict[output_name], output_data[idx])))

def generate_golden_data_run(args):
    import os
    import time
    import onnx
    import onnxruntime as ort
    import numpy as np
    from onnx import numpy_helper
    from node_utils import INPUT_TYPE_TO_NP_TYPE_MAP
    
    def save_input_output_data(save_path, data_dict, isInput=True):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        prefix = 'input' if isInput else 'output'
        for idx, (name, data) in enumerate(data_dict.items()):
            with open(os.path.join(save_path, f'{prefix}_{idx}.pb'), 'wb') as f:
                f.write(numpy_helper.from_array(data, name).SerializeToString())

    model = onnx.load(args.input_model)

    orig_shapes = {}
    for vi in model.graph.input:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else None)
        orig_shapes[vi.name] = dims

    so = ort.SessionOptions()
    so.register_custom_ops_library("libcustom_op_library.so")
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    providers = ['ROCMExecutionProvider']

    t0 = time.time()
    session = ort.InferenceSession(model.SerializeToString(),
                                   sess_options=so,
                                   providers=providers)
    t1 = time.time()
    print(f"Initialize ROCM session cost {(t1-t0)*1000:.2f} ms")

    input_dict = {}
    for inp in session.get_inputs():
        name = inp.name
        dtype_str = inp.type
        shape = []
        for d in orig_shapes[name]:
            if d is None:
                shape.append(args.batchsize)
            else:
                shape.append(d)
        print(f"[INFO] {name} <- shape {shape}, type={dtype_str}")

        data = np.random.rand(*shape)
        if 'uint8' in dtype_str:
            data = data * 255
        elif 'int8' in dtype_str:
            data = data * 255 - 128
        data = data.astype(INPUT_TYPE_TO_NP_TYPE_MAP[dtype_str])
        input_dict[name] = data

    if args.saveIOdata == 1:
        save_input_output_data(args.datapath, input_dict, isInput=True)

    output_names = [o.name for o in session.get_outputs()]

    # 6. Warm-up
    if args.warmup > 0:
        for _ in range(args.warmup):
            session.run(output_names, input_dict)

    t_start = time.time()
    for _ in range(args.runnum):
        outputs = session.run(output_names, input_dict)
    t_end = time.time()

    latency_ms = (t_end - t_start) * 1000 / (args.runnum * args.batchsize)
    print(f"Inference cost per sample: {latency_ms:.3f} ms  |  FPS: {1000/latency_ms:.2f}")

    if args.saveIOdata == 1:
        out_dict = {n: o for n, o in zip(output_names, outputs)}
        save_input_output_data(args.datapath, out_dict, isInput=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model",
                        type=str,
                        required=True,
                        default="",
                        help="input model file")
    parser.add_argument("-b", "--batchsize",
                        type=int,
                        required=False,
                        default=1,
                        help="batchsize")
    parser.add_argument("-c", "--checkresult",
                        type=bool,
                        required=False,
                        default=False,
                        help="check output accuracy")
    parser.add_argument("-d", "--datapath",
                        type=str,
                        required=True,
                        help="data path for saving golden data or checking output accuracy")
    parser.add_argument("-w", "--warmup",
                        type=int,
                        required=False,
                        default=50,
                        help="input warm up iterations")
    parser.add_argument("-n", "--runnum",
                        type=int,
                        required=False,
                        default=100,
                        help="input run model iterations")
    parser.add_argument("-s", "--inputshape",
                        type=int,
                        required=False,
                        default=-1,
                        help="bert input shape")
    parser.add_argument("-t", "--saveIOdata",
                        type=int,
                        required=False,
                        default=1,
                        help="save golden data")
    ARGS = parser.parse_args()

    if ARGS.checkresult:
        accuracy_check_run(ARGS)
        generate_golden_data_run(ARGS)
    else:
        generate_golden_data_run(ARGS)
