import numpy as np
import onnx_graphsurgeon as gs
import onnx
import torch
from utils import connectivity
import argparse

def main():
    parser = argparse.ArgumentParser(description='Removes quantization preceding residual connections')
    parser.add_argument('-i', '--input', type=str, help='input file name')
    parser.add_argument('-o', '--output', type=str, help='output file name')
    args = parser.parse_args()

    graph = gs.import_onnx(onnx.load(args.input))
    input_to_nodes, output_to_nodes, _, tensor_name_to_tensor = connectivity(graph)

    features_to_prune = {}
    floating_bias = {tensor_name: 0. for tensor_name in tensor_name_to_tensor.keys()}

    # Updates the dictionary features_to_prune by propagating the pruning of tensor downstream
    def propagate_pruning_downstream(tensor_name):
        for node_id in input_to_nodes[tensor_name]:
            node = graph.nodes[node_id]
            if node.op in ["BatchNormalization", "Cast", "MaxPool", "GlobalAveragePool", "Relu", "Reshape", "QuantizeLinear", "DequantizeLinear"]:
                for out in node.outputs:
                    features_to_prune[out.name] = features_to_prune[tensor_name]
                    propagate_pruning_downstream(out.name)

    def propagate_bias_downstream(tensor_name):
        for node_id in input_to_nodes[tensor_name]:
            node = graph.nodes[node_id]
            if node.op in ["BatchNormalization", "Cast", "MaxPool", "GlobalAveragePool", "Relu", "Add", "Reshape", "QuantizeLinear", "DequantizeLinear"]:
                for out in node.outputs:
                    if node.op == "Relu":
                        floating_bias[out.name] += floating_bias[tensor_name] * (floating_bias[tensor_name] > 0.0).astype(floating_bias[tensor_name].dtype)
                    else:
                        floating_bias[out.name] += floating_bias[tensor_name]
                    propagate_bias_downstream(out.name)
                floating_bias[tensor_name] = 0.

    # Updates the dictionary features_to_prune by propagating the pruning of tensor upstream
    def propagate_pruning_upstream(tensor_name, remove=False):
        for node_id in output_to_nodes[tensor_name]:
            node = graph.nodes[node_id]
            if node.op in ["BatchNormalization", "Cast", "MaxPool", "GlobalAveragePool", "Relu", "Add", "Reshape", "QuantizeLinear", "DequantizeLinear"]:
                for inp in node.inputs:
                    if inp.name in features_to_prune:
                        if remove:
                            del features_to_prune[inp.name]
                        else:
                            features_to_prune[inp.name] = features_to_prune[tensor_name]
                        propagate_pruning_upstream(inp.name, remove)

    # Performs the actual pruning of input features on a conv node
    def prune_input_features_conv(node, features):
        n_input_features = node.inputs[1].shape[1]
        features_to_keep = np.ones((n_input_features,), dtype=bool)
        features_to_keep[features] = False
        if type(floating_bias[node.inputs[0].name]) is not float:
            kh, kw = node.attrs["kernel_shape"]
            w = torch.tensor(node.inputs[1].values[:, np.logical_not(features_to_keep), :, :])
            inp = np.reshape(floating_bias[node.inputs[0].name], (1, -1, 1, 1))
            inp = np.tile(inp, (1, 1, kh, kw))
            inp = torch.tensor(inp)
            out = torch.nn.functional.conv2d(inp, w).numpy().flatten()
            if len(node.inputs) == 3:
                node.inputs[2].values += out
            else:
                bias = gs.Constant(name=node.name+".bias", values=out)
                node.inputs.append(bias)

        node.inputs[1].values = node.inputs[1].values[:, features_to_keep, :, :]

    # Performs the actual pruning of output features on a conv node
    def prune_output_features_conv(node, features):
        n_output_features = node.inputs[1].shape[0]
        features_to_keep = np.ones((n_output_features,), dtype=bool)
        features_to_keep[features] = False
        node.inputs[1].values = node.inputs[1].values[features_to_keep]
        if len(node.inputs) == 3:
            floating_bias[node.outputs[0].name] = node.inputs[2].values[np.logical_not(features_to_keep)]
            propagate_bias_downstream(node.outputs[0].name)
            node.inputs[2].values = node.inputs[2].values[features_to_keep]

    # Performs the actual pruning of input features on a qlinearconv node
    def prune_input_features_qlinearconv(node, features):
        n_input_features = node.inputs[3].shape[1]
        features_to_keep = np.ones((n_input_features,), dtype=bool)
        features_to_keep[features] = False
        node.inputs[3].values = node.inputs[3].values[:, features_to_keep, :, :]

    # Performs the actual pruning of output features on a qlinearconv node
    def prune_output_features_qlinearconv(node, features):
        n_output_features = node.inputs[3].shape[0]
        features_to_keep = np.ones((n_output_features,), dtype=bool)
        features_to_keep[features] = False
        node.inputs[3].values = node.inputs[3].values[features_to_keep]
        if len(node.inputs) == 9:
            node.inputs[8].values = node.inputs[8].values[features_to_keep]

    # Performs the actual pruning of input features on a gemm node
    def prune_input_features_gemm(node, features):
        n_input_features = node.inputs[1].shape[1]
        features_to_keep = np.ones((n_input_features,), dtype=bool)
        features_to_keep[features] = False
        node.inputs[1].values = node.inputs[1].values[:, features_to_keep]

    # Performs the actual pruning of features on a batchnormalization node
    def prune_features_bn(node, features):
        n_features = node.inputs[1].shape[0]
        features_to_keep = np.ones((n_features,), dtype=bool)
        features_to_keep[features] = False
        eps = node.attrs["epsilon"]
        effective_bias = node.inputs[2].values - node.inputs[1].values * node.inputs[3].values / np.sqrt(node.inputs[4].values + eps)
        floating_bias[node.outputs[0].name] = effective_bias[np.logical_not(features_to_keep)]
        propagate_bias_downstream(node.outputs[0].name)
        node.inputs[1].values = node.inputs[1].values[features_to_keep]
        node.inputs[2].values = node.inputs[2].values[features_to_keep]
        node.inputs[3].values = node.inputs[3].values[features_to_keep]
        node.inputs[4].values = node.inputs[4].values[features_to_keep]

    # Checks the prunable features across residual connections for consistency.
    # Determine consistent features to prune and propagates.
    def propagate_from_add(ignore_not_pruned=True):
        changed = False
        for node in graph.nodes:
            if node.op == "Add" and node.outputs[0].name not in features_to_prune:
                if node.inputs[0].name in features_to_prune and node.inputs[1].name in features_to_prune:
                    features_to_prune0 = features_to_prune[node.inputs[0].name]
                    features_to_prune1 = features_to_prune[node.inputs[1].name]
                    _features_to_prune = np.intersect1d(features_to_prune0, features_to_prune1)
                    if _features_to_prune.size > 0:
                        features_to_prune[node.inputs[0].name] = _features_to_prune
                        propagate_pruning_upstream(node.inputs[0].name)
                        features_to_prune[node.inputs[1].name] = _features_to_prune
                        propagate_pruning_upstream(node.inputs[1].name)
                        features_to_prune[node.outputs[0].name] = _features_to_prune
                        propagate_pruning_downstream(node.outputs[0].name)
                        changed = True
                    else:
                        del features_to_prune[node.inputs[0].name]
                        propagate_pruning_upstream(node.inputs[0].name, remove=True)
                        del features_to_prune[node.inputs[1].name]
                        propagate_pruning_upstream(node.inputs[1].name, remove=True)
                elif not ignore_not_pruned:
                    if node.inputs[0].name in features_to_prune:
                        del features_to_prune[node.inputs[0].name]
                        propagate_pruning_upstream(node.inputs[0].name, remove=True)
                    elif node.inputs[1].name in features_to_prune:
                        del features_to_prune[node.inputs[1].name]
                        propagate_pruning_upstream(node.inputs[1].name, remove=True)

        return changed


    # ------------
    # Pruning can only "emerge" from convolutional layers.
    # Cannot assess pruning from the layer alone since pruning can only happen if
    # there are no conflicts of pruned features across residual connections.
    # Loop through all nodes.
    # If convolutional, register tentative features to be pruned per output tensor.
    # Propagate the tentative pruning downstream until it reaches another
    # convolutional layer or residual connection.
    # ------------
    total_features = 0
    for node in graph.nodes:
        if node.op == "Conv":
            w = node.inputs[1].values
            n_features = w.shape[0]
            total_features += n_features
            zero_features = np.reshape(w, (n_features, -1))
            _features_to_prune = np.argwhere(np.amax(np.abs(zero_features), axis=-1).flatten() == 0.0).flatten()
            if _features_to_prune.size > 0:
                features_to_prune[node.outputs[0].name] = _features_to_prune
                propagate_pruning_downstream(node.outputs[0].name)
        elif node.op == "QLinearConv":
            w = node.inputs[3].values.astype(int) - node.inputs[5].values.astype(int)
            n_features = w.shape[0]
            total_features += n_features
            zero_features = np.reshape(w, (n_features, -1))
            _features_to_prune = np.argwhere(np.amax(np.abs(zero_features), axis=-1) == 0.0).flatten()
            if _features_to_prune.size > 0:
                features_to_prune[node.outputs[0].name] = _features_to_prune
                propagate_pruning_downstream(node.outputs[0].name)


    # ------------
    # Now check if consistency of prunable features across residual connections.
    # Repeat as long as the list of prunable features keeps changing.
    # Loop through all nodes. If "Add" node, check the prunable features from
    # both sides.
    # Make prunable features consistent and propagate them.
    # ------------
    changed = True
    while changed:
        changed = propagate_from_add()

    # ------------
    # Final loop through residual connections.
    # This fixes residual connections in which only one side had potential pruning.
    # ------------
    propagate_from_add(ignore_not_pruned=False)

    # Now we can be certain that the prunable features are consistent.
    # Prune features
    total_pruned_features = 0

    for tensor_name in features_to_prune:
        _features_to_prune = features_to_prune[tensor_name]
        tensor = tensor_name_to_tensor[tensor_name]
        if tensor.shape is not None:
            tensor.shape[-1] = (tensor.shape[-1] - _features_to_prune.size)
        for node_id in output_to_nodes[tensor_name]:
            node = graph.nodes[node_id]
            if node.op == "Conv":
                total_pruned_features += _features_to_prune.size
                prune_output_features_conv(node, _features_to_prune)
            elif node.op == "QLinearConv":
                total_pruned_features += _features_to_prune.size
                prune_output_features_qlinearconv(node, _features_to_prune)

    for tensor_name in features_to_prune:
        _features_to_prune = features_to_prune[tensor_name]
        for node_id in input_to_nodes[tensor_name]:
            node = graph.nodes[node_id]
            if node.op == "Conv":
                prune_input_features_conv(node, _features_to_prune)
            elif node.op == "QLinearConv":
                prune_input_features_qlinearconv(node, _features_to_prune)
            elif node.op == "BatchNormalization":
                prune_features_bn(node, _features_to_prune)
            elif node.op == "Gemm":
                prune_input_features_gemm(node, _features_to_prune)

    # Final cleanup and saving pruned graph
    graph.cleanup().toposort()
    onnx_graph = gs.export_onnx(graph)
    onnx.save(onnx_graph, args.output)

    print(f"Pruned features: {total_pruned_features}/{total_features}, {100.*total_pruned_features / total_features:.2f}%")

if __name__ == "__main__":
    main()

