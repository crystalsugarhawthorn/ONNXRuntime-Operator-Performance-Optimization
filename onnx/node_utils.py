import onnx
from onnx import shape_inference, helper, TensorProto
import numpy as np
import os
import fp16

DEBUG = False
INPUT_TYPE_TO_NP_TYPE_MAP = {
    'tensor(float)':      np.dtype('float32'),
    'tensor(uint8)':      np.dtype('uint8'),
    'tensor(int8)':       np.dtype('int8'),
    'tensor(uint16)':     np.dtype('uint16'),
    'tensor(int16)':      np.dtype('int16'),
    'tensor(int32)':      np.dtype('int32'),
    'tensor(int64)':      np.dtype('int64'),
    'tensor(bool)':       np.dtype('bool'),
    'tensor(float16)':    np.dtype('float16'),
    'tensor(float64)':    np.dtype('float64'),
    'tensor(complex64)':  np.dtype('complex64'),
    'tensor(complex128)': np.dtype('complex128'),
    'tensor(string)':     np.dtype(np.str_),
    'tensor(float8e5m2)':    np.dtype('int8'),
    'tensor(float8e5m2fnuz)': np.dtype('int8'),
    'tensor(float8e4m3fnuz)': np.dtype('int8'),
    'tensor(float8e4m3fn)':   np.dtype('int8'),
    'seq(tensor(complex64))':  np.dtype('complex64'),
    'seq(tensor(complex128))': np.dtype('complex128'),
    'seq(tensor(uint8))':      np.dtype('uint8'),
    'seq(tensor(int8))':       np.dtype('int8'),
    'seq(tensor(int16))':      np.dtype('int16'),
    'seq(tensor(uint16))':     np.dtype('uint16'),
    'seq(tensor(int32))':      np.dtype('int32'),
    'seq(tensor(uint32))':     np.dtype('uint32'),
    'seq(tensor(int64))':      np.dtype('int64'),
    'seq(tensor(uint64))':     np.dtype('uint64'),
    'seq(tensor(float))':      np.dtype('float32'),
    'seq(tensor(float16))':    np.dtype('float16'),
    'seq(tensor(double))':     np.dtype('double'),
    'seq(tensor(bool))':       np.dtype('bool'),
    'seq(tensor(string))':     np.dtype(np.str_)
}
NP_TYPE_TO_ONNX_TYPE_MAP = {
    np.dtype('float32') : 1,
    np.dtype('uint8') : 2,
    np.dtype('int8') : 3,
    np.dtype('uint16') : 4,
    np.dtype('int16') : 5,
    np.dtype('int32') : 6,
    np.dtype('int64') : 7,
    np.dtype(np.str_) : 8,
    np.dtype('bool') : 9,
    np.dtype('float16') : 10,
    np.dtype('float64') : 11,
    np.dtype('uint32') : 12,
    np.dtype('uint64') : 13,
    np.dtype('complex64') : 14,
    np.dtype('complex128') : 15,
}
Attribute_TYPE_MAP = {
    0 : 'UNDEFINED',
    1 : 'f',
    2 : 'i',
    3 : 's',
    4 : 't',
    5 : 'g',
    6 : 'floats',
    7 : 'ints',
    8 : 'strings',
    9 : 'tensors',
    10 : 'graphs'
}

class node_utils:
    def __init__(self, modelPath, batchSize=1, isINT8Model=True, inputShape=None):
        print('Process {}'.format(modelPath))
        self.modelPath = modelPath
        self.batchSize = batchSize
        self.isINT8Model = isINT8Model
        self.filter_op_type = ['QuantizeLinear', 'DequantizeLinear', 'Squeeze', 'Unsqueeze', 'Shape']
        self.select_input_op_type = ['Shape', 'Squeeze', 'Unsqueeze']
        self.quant_op_list = ['Conv', 'ConvTranspose', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Add', 'Sub', 'Div', 'Mul',
                              'MatMul', 'Transpose', 'Reshape', 'Flatten', 'Soft', 'Pad', 'Pow', 'Concat', 'LeakyRelu', 'Relu',
                              'PRelu', 'Clip', 'Resize', 'AveragePool', 'GlobalMaxPool', 'Split', 'Slice', 'Sigmoid',
                              'ReduceMean', 'Softplus', 'Tanh', 'ReduceSum', 'Gather', 'Swish', 'HardSigmoid', 'Mish']
        self.model_op_list = ['Swish','LayerNorm','Mish']
        self.input_node_dict = {}
        self.output_node_dict = {}
        self.name_node_dict = {}
        self.initializer_dict = {}
        self.value_info_dict = {}
        self.node_type_dict = {}
        self.model = onnx.load(modelPath)
        if inputShape is not None:
            for input in self.model.graph.input:
                if input.name not in inputShape:
                    print('[ERROR]: input {} with no input shape, please check!'.format(input.name))
                    exit()
                shape = inputShape[input.name]
                for idx, s in enumerate(shape):
                    input.type.tensor_type.shape.dim[idx].dim_value = s

        self.initModelState()

    def RefreshState(self):
        self.input_node_dict = {}
        self.output_node_dict = {}
        self.name_node_dict = {}
        self.initializer_dict = {}
        self.value_info_dict = {}
        self.initModelState()

    def initModelState(self):
        model = self.model
        for node in model.graph.node:
            if not node.name in self.name_node_dict:
                self.name_node_dict[node.name] = node
            for input in node.input:
                if not input in self.input_node_dict:
                    self.input_node_dict[input] = [node]
                else:
                    self.input_node_dict[input].append(node)
            for output in node.output:
                if not output in self.output_node_dict:
                    self.output_node_dict[output] = [node]
                else:
                    self.output_node_dict[output].append(node)

        for value_info in model.graph.value_info:
            self.value_info_dict[value_info.name] = value_info

        for initializer in model.graph.initializer:
            self.initializer_dict[initializer.name] = initializer

    def get_relate_qdq_node(self, name_list, is_front=True):
        qdq_list = []
        for name in name_list:
            if is_front and name in self.output_node_dict:
                    front_node_list = self.output_node_dict[name]
                    for node in front_node_list:
                        if node.op_type == 'DequantizeLinear':
                            qdq_list.append(node)
            elif not is_front and name in self.input_node_dict:
                back_node_list = self.input_node_dict[name]
                for node in back_node_list:
                    if node.op_type == 'QuantizeLinear':
                        qdq_list.append(node)
        return qdq_list

    def correct_value_info_by_ort(self):
        import onnxruntime as ort
        import time
        import copy

        def setBatchSize(model, batchSize):
            for input in model.graph.input:
                input.type.tensor_type.shape.dim[0].dim_value = batchSize
            for output in model.graph.output:
                output.type.tensor_type.shape.dim[0].dim_value = batchSize
            return model

        def needORTInfer(value_info):
            if len(value_info.type.tensor_type.shape.dim) == 0:
                return True

            if self.firstDimIsBatch(value_info) and value_info.type.tensor_type.shape.dim[0].dim_value == -1:
                return False
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.dim_value <= 0:
                    return True
            return False

        def canInferShapeByCPU(opset_import):
            for detail in opset_import:
                if detail.domain not in ['', 'ai.onnx', 'com.microsoft']:
                    return False
            return True

        def canInferShapeByCPU(opset_import):
            for detail in opset_import:
                if detail.domain not in ['', 'ai.onnx', 'com.microsoft']:
                    return False
            return True

        def firstDimIsBatch(value_info):
            node = self.output_node_dict[value_info.name][0]
            if node.op_type == 'Concat' and len(value_info.type.tensor_type.shape.dim) == 1:
                return False
            return True

        model = self.model
        model = setBatchSize(model, self.batchSize)
        if canInferShapeByCPU(model.opset_import):
            try:
                del model.graph.value_info[:]
                if self.modelPath.find('bert') < 0:
                    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
                else:
                    print("skip shape inference for bert...")
                self.model = model
                self.RefreshState()
            except:
                print("shape inference failed use onnxruntime infer later")

        batchsize = self.batchSize

        output_backup = copy.deepcopy(model.graph.output)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        if self.modelPath.find('bert') < 0 and canInferShapeByCPU(model.opset_import):
            EP_list = ['CPUExecutionProvider']
        else:
            EP_list = ['CUDAExecutionProvider']

        all_tensor_dict = {}
        for node in model.graph.node:
            for output in node.output:
                if output not in all_tensor_dict:
                    all_tensor_dict[output] = 1
                else:
                    all_tensor_dict[output] += 1
        value_info_list = []
        value_info_dict = {}
        value_info_list_back = []
        for tensor_name in list(all_tensor_dict.keys()):
            if DEBUG:
                print(tensor_name)
            if tensor_name not in self.value_info_dict:
                if DEBUG:
                    print('tensor {} not in self.value_info_dict and need infer by ORT'.format(tensor_name))
                value_info_list.append(tensor_name)
                value_info_dict[tensor_name] = onnx.ValueInfoProto(name=tensor_name)
                continue

            value_info = self.value_info_dict[tensor_name]
            if self.output_node_dict[tensor_name][0].op_type in self.filter_shape_correct_op_type:
                value_info_list_back.append(copy.deepcopy(value_info))
                if DEBUG:
                    print('add tensor {} to backup list'.format(tensor_name))
                continue

            if needORTInfer(value_info):
                value_info_list.append(tensor_name)
                value_info_dict[tensor_name] = value_info
                if DEBUG:
                    print('tensor {} need infer by ORT'.format(tensor_name))
            else:
                if len(value_info.type.tensor_type.shape.dim) > 0 and self.firstDimIsBatch(value_info):
                    value_info.type.tensor_type.shape.dim[0].dim_value = batchsize
                value_info_list_back.append(copy.deepcopy(value_info))
                if DEBUG:
                    print('modify tensor {} batchsize and add to backup list'.format(tensor_name))
        if len(value_info_list) > 0:
            step = 20
            infer_times = int(len(value_info_list) / step) if len(value_info_list) % step == 0 else int((len(value_info_list) + (step - len(value_info_list) % step)) / step)
            if DEBUG:
                print('infer_times %d' % infer_times)

            for i in range(infer_times):
                output_name_list = []
                if i == infer_times - 1:
                    output_name_list = value_info_list[step*i:]
                else:
                    output_name_list = value_info_list[step*i:step*(i+1)]

                if DEBUG:
                    print(i, output_name_list)
                del model.graph.output[:]
                for output_name in output_name_list:
                    model.graph.output.extend([onnx.ValueInfoProto(name=output_name)])

                del model.graph.value_info[:]
                start = time.time()
                cuda_session = ort.InferenceSession(model.SerializeToString(), providers=EP_list)
                end = time.time()
                duration = (end - start) * 1000
                if DEBUG:
                    print("Initialize Session cost {} ms\n".format(duration))

                inputs = cuda_session.get_inputs()

                input_dict = {}
                for input in inputs:
                    shape = []
                    for s in input.shape:
                        if isinstance(s, int):
                            shape.append(s)
                        else:
                            print('[ERROR]: input shape must be inter but input is {}'.format(type(s)))
                            exit()
                    for idx in range(len(shape)):
                        if shape[idx] is None:
                            shape[idx] = 1
                    input_data = np.random.random(shape)
                    if input.type != 0:
                        input_data = input_data*10
                    input_data = input_data.astype(INPUT_TYPE_TO_NP_TYPE_MAP[input.type])
                    input_dict[input.name] = input_data

                outputs = cuda_session.run(output_name_list, input_dict)
                for idx, output_name in enumerate(output_name_list):
                    if DEBUG:
                        print('tensor {} ort infered shape is {}'.format(output_name, outputs[idx].shape))
                    infered_shape = list(outputs[idx].shape)
                    new_tmp_value_info = helper.make_tensor_value_info(output_name, NP_TYPE_TO_ONNX_TYPE_MAP[outputs[idx].dtype], infered_shape)
                    value_info_dict[output_name] = new_tmp_value_info
                del cuda_session

            del model.graph.value_info[:]
            model.graph.value_info.extend(list(value_info_dict.values()))
            model.graph.value_info.extend(value_info_list_back)
            del model.graph.output[:]
            model.graph.output.extend(output_backup)
        else:
            del value_info_list_back

        self.model = setBatchSize(model, batchsize)

    def get_node_unique_info(self, node):
        unique_str = '{}-'.format(node.op_type)
        for input in node.input:
            if input in self.value_info_dict:
                value_info = self.value_info_dict[input]
                start_idx = 0
                for dim in value_info.type.tensor_type.shape.dim[start_idx:]:
                    unique_str += '{},'.format(dim.dim_value)
                unique_str = unique_str[:-1]
                unique_str += '_elemType_{}_'.format(value_info.type.tensor_type.elem_type)
            elif input in self.initializer_dict:
                initial_tensor = self.initializer_dict[input]
                start_idx = 0
                for dim in initial_tensor.dims[start_idx:]:
                    unique_str += '{},'.format(dim)
                unique_str = unique_str[:-1]
                unique_str += '_elemType_{}_'.format(initial_tensor.data_type)
            else:
                continue

        unique_str = unique_str[:-1]

        for attr in node.attribute:
            unique_str += '_{}_'.format(attr.name)
            attr_type = Attribute_TYPE_MAP[attr.type]
            value = getattr(attr, attr_type)
            if attr.type in [1, 2, 3]:
                unique_str += '{}'.format(value)
            elif attr.type in [6, 7, 8]:
                for v in value:
                    unique_str += '{},'.format(v)
                unique_str = unique_str[:-1]

        return unique_str

    def modify_FP32_to_FP16(self, graph):
        for input in graph.input:
            if input.type.tensor_type.elem_type == 1:
                input.type.tensor_type.elem_type = 10
        for output in graph.output:
            if output.type.tensor_type.elem_type == 1:
                output.type.tensor_type.elem_type = 10

        new_initializer = []
        for initializer in graph.initializer:
            if initializer.data_type == 1:
                new_initializer.append(fp16.convert_tensor_float_to_float16(initializer))
            else:
                new_initializer.append(initializer)

        del graph.initializer[:]
        graph.initializer.extend(new_initializer)
        return graph

    def extractNodeAndSave(self, save_path):
        self.correct_value_info_by_ort()
        self.RefreshState()
        model = self.model
        model_list = []
        node_list = []
        for node in model.graph.node:
            if node.op_type in self.filter_op_type:
                continue
            node_list.append(node)

        for idx, node in enumerate(node_list):
            if DEBUG:
                print(node.name, node.op_type)
            unique_str = self.get_node_unique_info(node=node)
            if unique_str not in self.node_type_dict:
                self.node_type_dict[unique_str] = [node.name]
            else:
                self.node_type_dict[unique_str].append(node.name)
                if DEBUG:
                   print('Repeated node, skip...')
                continue
            input_list = []
            output_list = []

            new_node_list = [node]
            new_initializer_list = []

            if self.isINT8Model:
                if node.op_type in self.quant_op_list:
                    dq_node = self.get_relate_qdq_node(node.input)
                    q_node = self.get_relate_qdq_node(node.output, False)
                    new_node_list.extend(dq_node)
                    new_node_list.extend(q_node)

            for new_node in new_node_list:
                input_list.extend(new_node.input)
                output_list.extend(new_node.output)

            inter_tensors = []
            for output in output_list:
                if output in input_list:
                    inter_tensors.append(output)

            pure_input_list = []
            pure_output_list = []
            for input in input_list:
                if input not in inter_tensors:
                    pure_input_list.append(input)

            for output in output_list:
                if output not in inter_tensors:
                    pure_output_list.append(output)

            new_graph_input = []
            for input in pure_input_list:
                if input in self.initializer_dict:
                    new_initializer_list.append(self.initializer_dict[input])
                else:
                    new_graph_input.append(input)
            graph_input_proto_list = []
            for graph_input in new_graph_input:
                if graph_input in self.value_info_dict:
                    input_value_info = self.value_info_dict[graph_input]
                else:
                    if DEBUG:
                        print('tensor {} not in self.value_info_dict'.format(graph_input))
                    for input in model.graph.input:
                        if input.name == graph_input:
                            input_value_info = input
                            if DEBUG:
                                print('get tensor {} from model.graph.input'.format(graph_input))
                graph_input_proto_list.append(input_value_info)

            graph_output_proto_list = []
            for graph_output in pure_output_list:
                if graph_output in self.value_info_dict:
                    output_value_info = self.value_info_dict[graph_output]
                else:
                    if DEBUG:
                        print('tensor {} not in self.value_info_dict'.format(graph_output))
                    for output in model.graph.output:
                        if output.name == graph_output:
                            output_value_info = output
                            if DEBUG:
                                print('get tensor {} from model.graph.output'.format(graph_output))
                graph_output_proto_list.append(output_value_info)

            if node.op_type in ['Reshape', 'Expand']:
                output_shape = [dim.dim_value for dim in graph_output_proto_list[0].type.tensor_type.shape.dim]
                Reshape_input_shape = helper.make_tensor(node.input[1], TensorProto.INT64, [len(output_shape)], np.array(output_shape))
                if len(graph_input_proto_list) == 2 and node.op_type == 'Reshape':
                    new_initializer_list.append(Reshape_input_shape)
                    remove_input_idx = 0
                    for input_idx in range(len(graph_input_proto_list)):
                        if graph_input_proto_list[input_idx].name == node.input[1]:
                            remove_input_idx = input_idx
                            break
                    del graph_input_proto_list[remove_input_idx]
                elif node.op_type == 'Expand':
                    new_initializer_list.append(Reshape_input_shape)
                    remove_input_idx = 0
                    for input_idx in range(len(graph_input_proto_list)):
                        if graph_input_proto_list[input_idx].name == node.input[0]:
                            remove_input_idx = input_idx
                            break
                    del graph_input_proto_list[remove_input_idx]

            if new_node_list[0].op_type == 'Cast':
                for input, output in zip(graph_input_proto_list, graph_output_proto_list):
                    assert len(input.type.tensor_type.shape.dim) == len(output.type.tensor_type.shape.dim)
                    for idx in range(len(input.type.tensor_type.shape.dim)):
                        if input.type.tensor_type.shape.dim[idx].dim_value != output.type.tensor_type.shape.dim[idx].dim_value:
                            input.type.tensor_type.shape.dim[idx].dim_value = output.type.tensor_type.shape.dim[idx].dim_value

            single_node_graph = helper.make_graph(
                new_node_list,
                '{}_single'.format(node.name),
                graph_input_proto_list,
                graph_output_proto_list,
                new_initializer_list
            )

            if len(new_node_list) == 1 and new_node_list[0].op_type != 'Cast':
                single_node_graph = self.modify_FP32_to_FP16(single_node_graph)

            if node.op_type in self.model_op_list:
                single_node_model = helper.make_model(
                    single_node_graph,
                    producer_name='model-single_op_model',
                    opset_imports=[helper.make_opsetid("", 13)]
                )
            else:
                single_node_model = helper.make_model(
                    single_node_graph,
                    producer_name='model-single_op_model',
                    opset_imports=[helper.make_opsetid("", 13)]
                )
            single_node_model.ir_version = model.ir_version

            try:
                single_node_model = shape_inference.infer_shapes(single_node_model)
            except:
                print('single node model infer shape error... save original model')

            new_path = '{}/{}/{}_{}.onnx'.format(save_path, node.op_type, node.op_type, idx)

            if not os.path.exists(os.path.split(new_path)[0]):
                os.makedirs(os.path.split(new_path)[0])
            onnx.save_model(single_node_model, new_path)
            model_name = os.path.basename(self.modelPath).split('.')[0]
            model_info = {"name": model_name, "path": new_path, "node_detail": unique_str}
            model_list.append(model_info)
        return model_list

if __name__ == '__main__':
    modelPath = 'models/modelzoo1.0/detection/ox_yolov4_dy/ox_yolov4_dy_int8.onnx'
    extractNode = extractNodefromModel(modelPath, 1)
    save_path = './yolov4_node_model'
    modelPathList, _ = extractNode.extractNodeAndSave(save_path)
    for nodeModelPath in modelPathList:
        print(nodeModelPath['name'], nodeModelPath['path'])
