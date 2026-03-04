
import textwrap
from typing import Any, Optional
import re
from pathlib import Path
import ast

# repo root
ROOT = Path(__file__).resolve().parents[2]

_DOTTED = re.compile(r"([A-Za-z_]\w*\.)+([A-Za-z_]\w*)")

def define_env(env):
    # Shared Mermaid header (init directive + styles)
    MERMAID_INIT = r'''
%%{init: {
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 20,
    "padding": 3,
    "curve": "linear"
  },
  "themeVariables": {
    "fontSize": "12px"
  }
}}%%
'''
    MERMAID_STYLE = r'''
  %% -------------------------
  %% Styles
  %% -------------------------
  classDef mod_main fill:#509e2f80,stroke:#203f13,stroke-width:2px,font-size:13,color:#203f13;
  classDef mod_mgmt fill:#6ad1e380,stroke:#125560,stroke-width:2px,font-size:13,color:#125560;
  classDef mod_opt  fill:#d9d9d680,stroke:#43433e,stroke-width:2px,font-size:13,color:#43433e;

  classDef data  fill:#fee8c880,stroke:#b30000,stroke-width:0.5px,font-size:11,color:#b30000;
  classDef param fill:#d7f4ee80,stroke:#165044,stroke-width:0.5px,font-size:11,color:#165044;

  classDef method   fill:#ffffff80,stroke:#000000,stroke-width:1px,font-size:12,color:#000000;
  classDef helper   fill:#f5f5f580,stroke:#616161,stroke-width:2px,font-size:12,color:#212121;
  classDef settings fill:#ffd5f680,stroke:#aa0088,stroke-width:1px,font-size:11,color:#aa0088;
'''

    @env.macro
    def mermaid_init() -> str:
        """
        Emit Mermaid shared init
        """
        return MERMAID_INIT
    
    @env.macro
    def mermaid_style() -> str:
        """
        Emit a Mermaid shared classDefs.
        Usage:
          {{ mermaid_style() }}
        """
        return MERMAID_STYLE
    
    @env.macro
    def docstring(obj: Any, file: Optional[str] = None) -> str:
        """
        Emit a py codeblock that shows Init signature and docstring of object,
        similar to Jupyter's `?` output.

        Parameters
        ----------
        obj:
            Either a Python object OR a dotted import path string, e.g.:
            "cibusmod.main_modules.regions.Regions"
        file:
            Optional override for the File: line (useful if you want to display
            a repo-relative path instead of an absolute path).
        """

        kind, node, class_name = _resolve_ast_obj(obj, file)

        sig = _signature_from_ast(kind, node, class_name)
        doc = _docstring_from_ast(node)

        # Assemble the output like a help() summary, but inside a code block
        lines = []
        lines.append(f"<b>Signature:</b>\n{sig}\n")
        lines.append("<b>Docstring:</b>")
        if doc:
            # Indent docstring body a bit to resemble help() formatting
            lines.append(textwrap.indent(doc, " " * 0))
        else:
            lines.append("    (No docstring.)")
        lines.append("")
        lines.append(f"<b>File:</b> {file}")

        body = "\n".join(lines).rstrip() + "\n"
        return f"<pre class='ds-pre'>\n{body}</pre>"

def _resolve_ast_obj(path: str, file: str):
    """
    Resolve a class/function/method from `file` using AST.

    Supports:
      - pkg.mod.Class
      - pkg.mod.func
      - pkg.mod.Class.method
      - ...method()  (trailing () ignored)

    Returns: (kind, node, class_name)
      kind: "class" | "function" | "method" | "init"
      node: ast node to extract docstring from
      class_name: str | None (used for init signature naming)
    """
    if path.endswith("()"):
        path = path[:-2]

    # Make file path absolute relative to repo root if needed
    fpath = Path(file)
    if not fpath.is_absolute():
        fpath = ROOT / fpath

    with open(fpath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(fpath))

    parts = path.split(".")
    last = parts[-1]

    # 1) module-level function?
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == last:
            return "function", node, None

    # 2) class?
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == last:
            # Prefer __init__ if present for signature, but docstring should be the class docstring
            init_node = None
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == "__init__":
                    init_node = sub
                    break
            # Return class node for docstring, init node for signature handled later
            # We'll return kind "class" and attach init_node via attribute
            node._ds_init_node = init_node  # harmless dynamic attr for our macro
            return "class", node, node.name

    # 3) method: ...Class.method
    if len(parts) >= 2:
        class_name = parts[-2]
        method_name = parts[-1]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
                        kind = "init" if method_name == "__init__" else "method"
                        return kind, sub, class_name

    raise ValueError(f"Object {path} not found in {file}")

def _docstring_from_ast(node):
    return ast.get_docstring(node) or ""

def _shorten_type(s: str) -> str:
    # Turn "CIBUSmod.x.y.Regions" -> "Regions" (also inside generics)
    return _DOTTED.sub(r"\2", s)

def _signature_from_ast(kind, node, class_name=None, width=50):
    """
    kind/node from _resolve_ast_obj.
    If kind == "class", node is ClassDef (docstring), and node._ds_init_node may exist.
    """
    # choose which function node to signature-print
    if kind == "class":
        init_node = getattr(node, "_ds_init_node", None)
        if init_node is None:
            return f"{class_name}()"
        func_node = init_node
        display_name = class_name
    elif kind == "init":
        func_node = node
        display_name = class_name
    else:
        func_node = node
        display_name = func_node.name

    params = []
    args = func_node.args

    # positional args
    for arg in args.args:
        if arg.arg in ("self", "cls"):
            continue
        ann = _shorten_type(ast.unparse(arg.annotation)) if arg.annotation else None
        params.append(f"{arg.arg}: {ann}" if ann else arg.arg)

    # varargs (*args)
    if args.vararg:
        ann = _shorten_type(ast.unparse(args.vararg.annotation)) if args.vararg.annotation else None
        params.append(f"*{args.vararg.arg}: {ann}" if ann else f"*{args.vararg.arg}")

    # kwonly args
    if args.kwonlyargs:
        if not args.vararg:
            params.append("*")
        for i, arg in enumerate(args.kwonlyargs):
            ann = _shorten_type(ast.unparse(arg.annotation)) if arg.annotation else None
            default = args.kw_defaults[i]
            part = f"{arg.arg}: {ann}" if ann else arg.arg
            if default is not None:
                part += f" = {_shorten_type(ast.unparse(default))}"
            params.append(part)

    # kwargs (**kwargs)
    if args.kwarg:
        ann = _shorten_type(ast.unparse(args.kwarg.annotation)) if args.kwarg.annotation else None
        params.append(f"**{args.kwarg.arg}: {ann}" if ann else f"**{args.kwarg.arg}")

    single = f"{display_name}({', '.join(params)})"
    if len(single) <= width:
        return single

    body = ",\n".join(f"    {p}" for p in params)
    return f"{display_name}(\n{body}\n)"