import onnx_graphsurgeon as gs
import onnx
from utils import replace_node
from conv import q_linear_conv_to_integer
import argparse
import numpy as np


def find_residual_quantization(graph):
    for node0 in graph.nodes:
        if node0.op == "Add":
            add_node = node0
            for node1 in graph.nodes:
                if node1.op == "QLinearConv":
                    conv_node = node1
                    dequant_node = None
                    for node2 in graph.nodes:
                        if node2.op == "DequantizeLinear" and conv_node.outputs[0] == node2.inputs[0]:
                            dequant_node = node2
                            break
                    if dequant_node is not None:
                        if dequant_node.outputs[0] in add_node.inputs:
                            return conv_node, dequant_node

    return None, None


def main():
    parser = argparse.ArgumentParser(description='Removes quantization preceding residual connections')
    parser.add_argument('-i', '--input', type=str, help='input file name')
    parser.add_argument('-o', '--output', type=str, help='output file name')
    args = parser.parse_args()

    graph = gs.import_onnx(onnx.load(args.input))

    conv_node, dequant_node = find_residual_quantization(graph)
    while conv_node is not None:
        x_scale = conv_node.inputs[1]
        w_scale = conv_node.inputs[4]
        y_scale = conv_node.inputs[6]
        y_zero_point = conv_node.inputs[7]
        B = None
        if len(conv_node.inputs) == 9:
            B = conv_node.inputs[8]

        dequant_scale = dequant_node.inputs[1]
        dequant_zero_point = dequant_node.inputs[2]
        dequant_scale.values = np.array(x_scale.values * w_scale.values, dtype=np.float32)
        dequant_zero_point.values = np.array(0, dtype=np.int32)
        q_linear_conv_to_integer(conv_node)

        if B:
            B.values = np.reshape(B.values, (1, -1, 1, 1))
            add_bias_inputs = [conv_node.outputs[0], B]
            add_bias_outputs = [gs.Variable(name=conv_node.name + "_addbias_output")]
            add_bias = gs.Node(op="Add", name=conv_node.name + "_addbias", inputs=add_bias_inputs, outputs=add_bias_outputs)
            dequant_node.inputs[0] = add_bias.outputs[0]
            graph.nodes.append(add_bias)

        '''
        quant_outputs = [gs.Variable(name=conv_node.name + "_quant_output")]
        dequant_outputs = [gs.Variable(name=conv_node.name + "_dequant_output")]
        quant = gs.Node(op="QuantizeLinear", name=conv_node.name + "_quant", inputs=dequant_node.outputs + [y_scale, y_zero_point], outputs=quant_outputs)
        dequant = gs.Node(op="DequantizeLinear", name=conv_node.name + "_dequant", inputs=quant_outputs + [y_scale, y_zero_point], outputs=dequant_outputs)

        for n in graph.nodes:
            if n.op == "Add" and dequant_node.outputs[0] in n.inputs:
                indx = n.inputs.index(dequant_node.outputs[0])
                n.inputs[indx] = dequant.outputs[0]
                break

        graph.nodes.append(quant)
        graph.nodes.append(dequant)
        '''

        conv_node, dequant_node = find_residual_quantization(graph)

    graph.cleanup().toposort()
    onnx_graph = gs.export_onnx(graph)
    onnx.save(onnx_graph, args.output)


if __name__ == "__main__":
    main()
