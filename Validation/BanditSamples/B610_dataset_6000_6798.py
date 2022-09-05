def _print_details(extra=None):
    """Return a function that prints node details."""
    def print_node_handler(name, node, depth):
        """Standard printer for a node."""
        line = "{0}{1} {2} ({3}:{4})".format(depth,
                                             (" " * depth),
                                             name,
                                             node.line,
                                             node.col)
        if extra is not None:
            line += " [{0}]".format(extra(node))

        sys.stdout.write(line + "\n")

    return print_node_handler