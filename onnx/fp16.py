import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

def _npfloat16_to_int(np_list):
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]

def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    def between(a, b, c):
        return np.logical_and(a < b, b < c)
    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)

def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data),
                                                 min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        if tensor.raw_data:
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            tensor.raw_data = float16_list.tostring()
    return tensor

def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)
