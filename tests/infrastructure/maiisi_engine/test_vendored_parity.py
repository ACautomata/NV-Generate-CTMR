"""Vendored parity guard (issue #134): the maiisi_engine freeze must not drift.

Compares ASTs of the vendored files against their ``scripts/`` originals with
every import node stripped — exactly the documented delta of vendoring
(package-home import rewrite). Any behavioral edit on either side breaks this
test, keeping the "byte-stable behavior" acceptance machine-checked. The
support modules are guarded function-by-function against their extraction
sources.

Pure text processing: runs on the light stack, no torch needed.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

ENGINE_PAIRS = [
    ("diff_model_setting.py", "diff_model_setting.py"),
    ("diff_model_train.py", "diff_model_train.py"),
    ("diff_model_infer.py", "diff_model_infer.py"),
    ("diff_model_create_training_data.py", "create_training_data.py"),
    ("utils_infer.py", "utils_infer.py"),
]

# (source file, source symbol) -> (vendored module, same symbol)
SUPPORT_SYMBOLS = [
    ("utils.py", "define_instance", "instance_definition.py"),
    ("transforms.py", "SUPPORT_MODALITIES", "instance_definition.py"),
    ("transforms.py", "define_fixed_intensity_transform", "instance_definition.py"),
    ("utils.py", "dynamic_infer", "inference_primitives.py"),
    ("utils.py", "get_body_region_index_from_mask", "inference_primitives.py"),
    ("sample_mask.py", "check_input_ct", "inference_primitives.py"),
    ("sample_mask.py", "check_input_mr", "inference_primitives.py"),
]


def _strip_import_nodes(tree: ast.AST) -> ast.AST:
    """Remove Import/ImportFrom nodes from every body in the tree."""
    for node in list(ast.walk(tree)):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [n for n in body if not isinstance(n, ast.Import | ast.ImportFrom)]
    return tree


def _dump_without_imports(path: Path) -> str:
    return ast.dump(_strip_import_nodes(ast.parse(path.read_text())))


def _top_level_defs(path: Path) -> dict[str, ast.AST]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node
            else:
                out[node.name] = node
    return out


def test_engine_files_match_originals_ast():
    for orig_name, vendored_name in ENGINE_PAIRS:
        original = REPO_ROOT / "scripts" / orig_name
        vendored = REPO_ROOT / "src" / "ctmr" / "infrastructure" / "maiisi_engine" / vendored_name
        assert _dump_without_imports(original) == _dump_without_imports(vendored), (
            f"{vendored_name} drifted from scripts/{orig_name}; engine freeze is byte-stable by contract — update both sides or write a new ADR."
        )


def test_module_docstrings_unchanged():
    """The only non-import edits allowed at module level are appended banner comments."""
    for orig_name, vendored_name in ENGINE_PAIRS:
        original_doc = ast.get_docstring(ast.parse((REPO_ROOT / "scripts" / orig_name).read_text()))
        vendored_doc = ast.get_docstring(ast.parse((REPO_ROOT / "src" / "ctmr" / "infrastructure" / "maiisi_engine" / vendored_name).read_text()))
        assert original_doc == vendored_doc, f"{vendored_name}: module docstring changed"


@pytest.mark.parametrize("src_name,symbol,module_name", SUPPORT_SYMBOLS)
def test_support_symbols_extracted_byte_stable(src_name: str, symbol: str, module_name: str):
    source_defs = _top_level_defs(REPO_ROOT / "scripts" / src_name)
    support_defs = _top_level_defs(REPO_ROOT / "src" / "ctmr" / "infrastructure" / "maiisi_engine" / module_name)
    assert symbol in source_defs, f"{src_name} no longer defines {symbol}"
    assert symbol in support_defs, f"{module_name} lost {symbol}"
    assert ast.dump(source_defs[symbol]) == ast.dump(support_defs[symbol]), f"{module_name}.{symbol} drifted from scripts/{src_name}"
