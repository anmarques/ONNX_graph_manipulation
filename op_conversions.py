import onnx_graphsurgeon as gs
import numpy as np


def q_linear_conv_to_integer(node):
    assert node.op == "QLinearConv"

    x = node.inputs[0]
    x_scale = node.inputs[1]
    x_zero_point = node.inputs[2]
    w = node.inputs[3]
    w_scale = node.inputs[4]
    w_zero_point = node.inputs[5]
    y_zero_point = node.inputs[7]
    use_bias = len(node.inputs) == 9

    if use_bias:
        bias = node.inputs[8]

    add_relu = y_zero_point.dtype is np.uint8

    quant_outputs = [gs.Variable(node.name + '_quant_output')]
    quant_node = gs.Node(name=node.name + '_quant', op="QuantizeLinear", inputs=[x, x_scale, x_zero_point],
                         outputs=quant_outputs)
    new_nodes = [quant_node]

    conv_outputs = [gs.Variable(node.name + '_conv_output')]
    cast_inputs = conv_outputs
    if use_bias:
        bias_outputs = [gs.Variable(node.name + '_bias_output')]
        bias.values = np.reshape(bias.values, (1, -1, 1, 1))
        bias_inputs = conv_outputs + [bias]
        bias_node = gs.Node(name=node.name + '_bias', op='Add', inputs=bias_inputs, outputs=bias_outputs)
        new_nodes.append(bias_node)
        cast_inputs = bias_outputs

    cast_outputs = [gs.Variable(node.name + '_cast_output')]
    cast_node = gs.Node(name=node.name + '_cast', op="Cast", attrs={"to": 1}, inputs=cast_inputs, outputs=cast_outputs)
    new_nodes.append(cast_node)

    mul_scale = gs.Constant(name=node.name + '_float_scale', values=np.array(x_scale.values * w_scale.values))
    mul_inputs = cast_outputs + [mul_scale]
    mul_node = gs.Node(name=node.name + '_mul', op="Mul", inputs=mul_inputs,
                       outputs=node.outputs)
    new_nodes.append(mul_node)

    if add_relu:
        mul_outputs = [gs.Variable(name=node.name + '_mul_output')]
        mul_node.outputs = mul_outputs
        relu_node = gs.Node(name=node.name + '_relu', op="Relu", inputs=mul_outputs,
                            outputs=node.outputs)
        new_nodes.append(relu_node)

    node.op = "ConvInteger"
    node.inputs = [quant_outputs[0], w, x_zero_point, w_zero_point]
    node.outputs = conv_outputs

    return new_nodes


def quantized_add(node, scales, zero_points, add_relu=False):
    assert node.op == "Add"

    new_nodes = []
    if scales[0] is not None:
        quant0_outputs = [gs.Variable(node.name + '_quant0_output')]
        quant0_node = gs.Node(name=node.name + '_quant0', op="QuantizeLinear",
                              inputs=[node.inputs[0], scales[0], zero_points[0]],
                              outputs=quant0_outputs)
        new_nodes.append(quant0_node)

        dequant0_outputs = [gs.Variable(node.name + '_dequant0_output')]
        dequant0_node = gs.Node(name=node.name + '_dequant0', op="DequantizeLinear",
                                inputs=[quant0_outputs[0], scales[0], zero_points[0]],
                                outputs=dequant0_outputs)
        new_nodes.append(dequant0_node)

        node.inputs[0] = dequant0_outputs[0]

    if scales[1] is not None:
        quant1_outputs = [gs.Variable(node.name + '_quant1_output')]
        quant1_node = gs.Node(name=node.name + '_quant1', op="QuantizeLinear",
                              inputs=[node.inputs[1], scales[1], zero_points[1]],
                              outputs=quant1_outputs)
        new_nodes.append(quant1_node)

        dequant1_outputs = [gs.Variable(node.name + '_dequant1_output')]
        dequant1_node = gs.Node(name=node.name + '_dequant1', op="DequantizeLinear",
                                inputs=[quant1_outputs[0], scales[1], zero_points[1]],
                                outputs=dequant1_outputs)
        new_nodes.append(dequant1_node)

        node.inputs[1] = dequant1_outputs[0]

    if add_relu:
        add_outputs = [gs.Variable(name=node.name + '_add_output')]
        relu_node = gs.Node(name=node.name + '_relu', op="Relu", inputs=add_outputs,
                            outputs=node.outputs)
        node.outputs = add_outputs
        new_nodes.append(relu_node)

    return new_nodes


def quantized_maxpool(node, scale, zero_point):
    assert node.op == "MaxPool"

    new_nodes = []
    if scale is not None:
        quant_outputs = [gs.Variable(node.name + '_quant_output')]
        quant_node = gs.Node(name=node.name + '_quant', op="QuantizeLinear",
                             inputs=[node.inputs[0], scale, zero_point],
                             outputs=quant_outputs)
        new_nodes.append(quant_node)

        dequant_outputs = [gs.Variable(node.name + '_dequant_output')]
        dequant_node = gs.Node(name=node.name + '_dequant', op="DequantizeLinear",
                               inputs=[quant_outputs[0], scale, zero_point],
                               outputs=dequant_outputs)
        new_nodes.append(dequant_node)

        node.inputs[0] = dequant_outputs[0]

    return new_nodes
