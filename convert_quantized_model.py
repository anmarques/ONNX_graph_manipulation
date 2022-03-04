import onnx_graphsurgeon as gs
import onnx
from utils import connectivity, remove_node, remove_duplicate, remove_pattern
from op_conversions import q_linear_conv_to_integer, quantized_add, quantized_maxpool
import argparse


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
    parser.add_argument('-f', '--fix_residual', action='store_true', help='perform residual fix')
    args = parser.parse_args()

    graph = gs.import_onnx(onnx.load(args.input))

    to_remove = []
    for node in graph.nodes:
        if node.op in ["QuantizeLinear", "DequantizeLinear"]:
            to_remove.append(node)

    new_nodes = []
    for node in graph.nodes:
        if node.op == "QLinearConv":
            new_nodes.extend(q_linear_conv_to_integer(node))
    graph.nodes.extend(new_nodes)

    input_to_nodes, output_to_nodes, node_name_to_id = connectivity(graph)

    new_nodes = []
    for node in graph.nodes:
        if node.op == "Add":
            if any([isinstance(inp, gs.Constant) for inp in node.inputs]):
                continue
            scales = []
            zero_points = []
            for inp in node.inputs:
                previous_node = graph.nodes[output_to_nodes[inp.name][0]]
                if previous_node.op in ["QuantizeLinear", "DequantizeLinear"]:
                    scales.append(previous_node.inputs[1])
                    zero_points.append(previous_node.inputs[2])
                else:
                    scales.append(None)
                    zero_points.append(None)
            new_nodes.extend(quantized_add(node, scales, zero_points, True))
        elif node.op == "MaxPool":
            previous_node = graph.nodes[output_to_nodes[node.inputs[0].name][0]]
            if previous_node.op in ["QuantizeLinear", "DequantizeLinear"]:
                scale = previous_node.inputs[1]
                zero_point = previous_node.inputs[2]
            else:
                scale = None
                zero_point = None
            new_nodes.extend(quantized_maxpool(node, scale, zero_point))

    graph.nodes.extend(new_nodes)

    for node in to_remove:
        remove_node(graph, node)
    remove_duplicate(graph, "Relu")
    remove_pattern(graph, ["Relu", "QuantizeLinear", "DequantizeLinear", "Add"], [0])
    if args.fix_residual:
        remove_pattern(graph, ["Mul", "Relu", "QuantizeLinear", "ConvInteger", "Add", "Cast", "Mul", "QuantizeLinear", "DequantizeLinear", "Add"], [7, 8])
    graph.cleanup().toposort()
    onnx_graph = gs.export_onnx(graph)
    onnx.save(onnx_graph, args.output)


if __name__ == "__main__":
    main()
