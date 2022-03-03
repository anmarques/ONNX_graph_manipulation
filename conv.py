import onnx_graphsurgeon as gs
import numpy as np
from copy import deepcopy


def q_linear_conv_to_integer(node):
    assert node.op == "QLinearConv"

    x = node.inputs[0]
    x_zero_point = node.inputs[2]
    w = node.inputs[3]
    w_zero_point = node.inputs[5]

    node.inputs = [x, w, x_zero_point, w_zero_point]
    node.op = "ConvInteger"
    node.outputs[0].type = np.int32
