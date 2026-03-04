from __future__ import annotations

import os
import sys
import importlib
import inspect
import textwrap
from typing import Any, Optional
from pathlib import Path
import html

# repo root
ROOT = Path(__file__).resolve().parents[2]

# add repo root so `import CIBUSmod` works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        target = _resolve_obj(obj)

        # Prefer showing constructor signature for classes, else call signature
        init_sig = _init_signature_string(target)

        # Docstring (cleaned like help())
        doc = inspect.getdoc(target) or ""
        doc = doc.rstrip()

        # File path (try to be robust)
        src_file = file or _safe_source_file(target)

        # Assemble the output like a help() summary, but inside a code block
        lines = []
        lines.append(f"<b>Signature:</b>\n{init_sig}\n")
        lines.append("<b>Docstring:</b>")
        if doc:
            # Indent docstring body a bit to resemble help() formatting
            lines.append(textwrap.indent(doc, " " * 0))
        else:
            lines.append("    (No docstring.)")
        lines.append("")
        lines.append(f"<b>File:</b> {src_file}")

        body = "\n".join(lines).rstrip() + "\n"
        return f"<pre class='ds-pre'>\n{body}</pre>"

def _resolve_obj(obj: Any) -> Any:
    if not isinstance(obj, str):
        return obj

    path = obj.strip()

    if path.endswith("()"):
        path = path[:-2]

    parts = path.split(".")

    # progressively try to import module
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            mod = importlib.import_module(mod_name)
            attr_parts = parts[i:]
            return _deep_getattr(mod, ".".join(attr_parts)) if attr_parts else mod
        except ImportError:
            continue

    raise ImportError(f"Could not resolve object: {path}")


def _deep_getattr(root: Any, attr_path: str) -> Any:
    cur = root
    for part in attr_path.split("."):
        cur = getattr(cur, part)
    return cur


def _init_signature_string(target: Any, width: int = 50) -> str:
    """
    For classes: show ClassName(<init params>).
    For callables: show name(<params>).
    Fall back gracefully if signature can't be obtained.
    """
    name = getattr(target, "__name__", target.__class__.__name__)

    # obtain signature
    try:
        if inspect.isclass(target):
            sig = inspect.signature(target.__init__)
            params = list(sig.parameters.values())

            # drop self / cls
            if params and params[0].name in ("self", "cls"):
                params = params[1:]

        else:
            sig = inspect.signature(target)
            params = list(sig.parameters.values())

    except (TypeError, ValueError):
        return f"{name}(*args, **kwargs)"

    param_strings = []

    for p in params:
        part = p.name

        ann = _short_annotation(p.annotation)
        if ann:
            part += f": {ann}"

        if p.default is not inspect._empty:
            part += f" = {p.default}"

        param_strings.append(part)

    single = f"{name}({', '.join(param_strings)})"
    if len(single) <= width:
        return single

    # multiline format
    indent = " " * 4
    body = ",\n".join(f"{indent}{p}" for p in param_strings)

    return f"{name}(\n{body}\n)"

def _safe_source_file(target: Any) -> str:
    """
    Try best-effort source file resolution; return '(unknown)' if not available.
    """
    try:
        f = inspect.getsourcefile(target) or inspect.getfile(target)
        if not f:
            return "(unknown)"
        # normalize path a bit
        return os.path.normpath(f)
    except Exception:
        return "(unknown)"
    
def _short_annotation(ann):
    """Return short type name for annotations."""
    if ann is inspect._empty:
        return None

    # class types
    if hasattr(ann, "__name__"):
        return ann.__name__

    # typing / forward refs like 'CIBUSmod...Regions'
    s = str(ann)

    # remove quotes
    s = s.strip("'")

    # keep only last dotted name
    if "." in s:
        s = s.split(".")[-1]

    return s