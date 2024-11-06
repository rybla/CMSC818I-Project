import ast

DEBUG = True


def debug_log(msg):
    if DEBUG:
        print(f"[>] {msg}")


def find_innermost_scope(tree, line_number: int):
    """
    Finds the innermost function or class definition that encloses a given line number.

    Args:
        tree: The AST of the Python code.
        line_number: The target line number.

    Returns:
        The AST node of the innermost scope, or None if not found.
    """

    innermost = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef)
        ) and all(
            [
                innermost.lineno < line_number if innermost is not None else True,
                (
                    (
                        line_number < innermost.end_lineno
                        if innermost.end_lineno is not None
                        else False
                    )
                    if innermost is not None
                    else True
                ),
                node.lineno < line_number,
                line_number < node.end_lineno if node.end_lineno is not None else False,
            ]
        ):
            innermost = node
    return innermost
