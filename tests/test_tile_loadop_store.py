import pytest
import graphics.tile_loadop_store as mod

CASES = [
    ("loadop_by_need", (1,)),
    ("resolve_outside", (1,)),
    ("skip_store_mid", (1,)),
    ("explicit_dependency", (1,)),
    ("readonly_depth", (1,)),
    ("merge_layout", (1,)),
    ("begin_pass", (1,)),
    ("end_pass", (1,)),
    ("resolve", (1,)),
    ("submit", (1,)),
    ("validate_pass", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
