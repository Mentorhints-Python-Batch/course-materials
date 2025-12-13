def map(graph: dict) -> dict:
    """
    Create a mapping of child nodes to their parent nodes.
    Args:
        graph (dict): A dictionary representing the tree structure where keys are parent nodes and values are lists of child nodes.
    Returns:
        dict: A dictionary mapping child nodes to their parent nodes.
    """
    map = {}
    for parent, children in graph.items():
        for child in children:
            map[child] = parent
    return map

def find_parent(node: int, graph: dict) -> int:
    """
    Find the parent of a given node in the tree structure.
    Args:
        node (int): The node whose parent is to be found.
        graph (dict): A dictionary representing the tree structure where keys are parent nodes and values are lists of child nodes.
    Returns:
        int: The parent node of the given node. If the node has no parent, returns -9999999.
    """
    p_c_map = map(graph)
    try:
        return p_c_map[node]
    except KeyError:
        return -9999999
