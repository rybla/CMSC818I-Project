import ast

DEBUG = True


def debug_log(msg):
    if DEBUG:
        print(f"[>] {msg}")


def find_innermost_scope(tree, line_range: tuple[int, int]):
    """
    Finds the innermost function or class definition that encloses a given line number.
    """

    innermost = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.stmt))
            and (
                node.lineno <= line_range[0]
                and (
                    line_range[1] <= node.end_lineno
                    if isinstance(node.end_lineno, int)
                    else True
                )
            )
            and (
                (
                    innermost.lineno < node.lineno
                    and node.end_lineno < innermost.end_lineno
                    if isinstance(node.end_lineno, int)
                    and isinstance(innermost.end_lineno, int)
                    else False
                )
                if innermost is not None
                else True
            )
        ):
            innermost = node
    return innermost if innermost is not None else tree
