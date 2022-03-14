def connectivity(graph):
    input_to_nodes = {}
    output_to_nodes = {}
    node_name_to_id = {}
    tensor_name_to_tensor = {}
    for node_id, node in enumerate(graph.nodes):
        node_name_to_id['node.name'] = node_id
        for var in node.inputs:
            if var.name in input_to_nodes:
                input_to_nodes[var.name].append(node_id)
            else:
                input_to_nodes[var.name] = [node_id]
            if var.name not in tensor_name_to_tensor:
                tensor_name_to_tensor[var.name] = var
        for var in node.outputs:
            if var.name in output_to_nodes:
                output_to_nodes[var.name].append(node_id)
            else:
                output_to_nodes[var.name] = [node_id]
            if var.name not in tensor_name_to_tensor:
                tensor_name_to_tensor[var.name] = var

    return input_to_nodes, output_to_nodes, node_name_to_id, tensor_name_to_tensor


def replace_node(graph, to_replace, new_nodes):
    assert to_replace in graph.nodes

    to_remove = None
    for node in graph.nodes:
        if to_replace == node:
            to_remove = node
            outputs_to_match = to_replace.outputs
            break

    if to_remove is not None:
        graph.nodes.remove(to_remove)
        for node in graph.nodes:
            for input_id, inp in enumerate(node.inputs):
                if inp in outputs_to_match:
                    output_id = outputs_to_match.index(inp)
                    node.inputs[input_id] = new_nodes[-1].outputs[output_id]

        graph.nodes.extend(new_nodes)

    return graph


def remove_duplicate(graph, op):
    input_to_nodes, _, _ = connectivity(graph)
    to_remove = []
    for node in graph.nodes:
        if node.op == op:
            for next_node_id in input_to_nodes[node.outputs[0].name]:
                next_node = graph.nodes[next_node_id]
                if next_node.op == op:
                    to_remove.append(next_node)

    for node in to_remove:
        remove_node(graph, node)


def remove_pattern(graph, pattern, index):
    input_to_nodes, _, _ = connectivity(graph)
    to_remove = []
    for node in graph.nodes:
        next_node = node
        pattern_nodes = []
        for op in pattern:
            if next_node.op == op:
                pattern_nodes.append(next_node)
                next_node_id = input_to_nodes[next_node.outputs[0].name][0]
                next_node = graph.nodes[next_node_id]
            else:
                break
        if len(pattern_nodes) == len(pattern):
            for i in index:
                to_remove.append(pattern_nodes[i])

    for node in to_remove:
        remove_node(graph, node)


def remove_node(graph, to_remove):
    assert to_remove in graph.nodes and len(to_remove.outputs) == 1

    for node in graph.nodes:
        if len(node.inputs) >= 1 and to_remove.outputs[0] in node.inputs:
            input_id = node.inputs.index(to_remove.outputs[0])
            node.inputs[input_id] = to_remove.inputs[0]

    graph.nodes.remove(to_remove)

    return graph
