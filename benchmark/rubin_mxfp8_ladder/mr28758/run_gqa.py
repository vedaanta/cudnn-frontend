#!/usr/bin/env python3
"""Run MR 28758's gqa.py on the installed (July 2026) internal CuTe DSL: shim the two APIs the MR's newer DSL added."""
import os, sys, runpy, types
R = os.path.dirname(os.path.abspath(__file__))
FMHA = os.path.join(R, "CuTeDSL", "blackwell", "kernel", "attention", "fmha")
sys.path[:0] = [os.path.join(R, "CuTeDSL"), FMHA]
import cutlass, cutlass.cute as cute
import cutlass.cute.nvgpu as nvgpu
import cutlass.cute.nvgpu.internal as internal
if not hasattr(internal, "ifence"):
    if os.environ.get("GQA_IFENCE", "0") == "1":
        from cutlass._mlir.dialects import llvm as _llvm
        def ifence(*, loc=None, ip=None):  # the MR's implementation: a ptxas knob pragma (interference fence)
            _llvm.inline_asm(res=None, operands_=[], asm_string='.pragma "next knob FenceInterference";\n', constraints="~{memory}",
                             has_side_effects=True, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        print("[shim] ifence -> .pragma FenceInterference (GQA_IFENCE=1)", flush=True)
    else:
        def ifence(*, loc=None, ip=None):  # interference fence = register-allocation hint only; no-op shim
            return None
        print("[shim] ifence -> no-op", flush=True)
    internal.ifence = ifence
    nvgpu.ifence = ifence
import cutlass.memory as cmem
if not hasattr(cmem, "get_smem_capacity_in_bytes"):
    def get_smem_capacity_in_bytes(arch):
        return int(os.environ.get("GQA_SMEM_BYTES", "232448"))  # B200 default; set for Rubin
    cmem.get_smem_capacity_in_bytes = get_smem_capacity_in_bytes
    print(f"[shim] get_smem_capacity_in_bytes -> {get_smem_capacity_in_bytes(None)}", flush=True)
import cutlass.utils.blackwell_helpers as bh
if not hasattr(bh, "make_smem_layout_c"):
    # port the one helper the MR-era DSL added, executed in the module's own namespace so its free names resolve
    src = open(os.path.join(R, "blackwell_helpers_new.py")).read()
    i_def = src.index("\ndef make_smem_layout_c(")
    j = src.index("\ndef ", i_def + 1)
    k2 = src.rfind("\n", 0, j)  # drop the NEXT function's decorator line
    if src[k2 + 1:j].strip().startswith("@"):
        j = k2
    i = i_def
    k = src.rfind("\n", 0, i_def)  # include a decorator line if present
    if src[k + 1:i_def].strip().startswith("@"):
        i = k
    exec(src[i:j], bh.__dict__)
    print("[shim] make_smem_layout_c ported from the MR-era blackwell_helpers", flush=True)
def _port(module, new_file, func_name, export_to=None):
    """Exec one function from the MR-era DSL source into the installed module's namespace (free names resolve there)."""
    src = open(os.path.join(R, new_file)).read()
    i_def = src.index("\ndef " + func_name + "(")
    j = src.index("\ndef ", i_def + 1)
    k2 = src.rfind("\n", 0, j)
    if src[k2 + 1:j].strip().startswith("@"):
        j = k2
    i = i_def
    k = src.rfind("\n", 0, i_def)
    if src[k + 1:i_def].strip().startswith("@"):
        i = k
    exec(src[i:j], module.__dict__)
    if export_to is not None:
        setattr(export_to, func_name, getattr(module, func_name))
    print(f"[shim] {func_name} ported from the MR-era DSL", flush=True)

import cutlass.cute.nvgpu.helpers as nvh
import cutlass.cute.nvgpu.cpasync.copy as _cpcopy
import cutlass.cute.typing as _cty
def _prefill_names(module, new_file):
    """Give the installed module every name the MR-era file imports from cpasync.copy / typing (placeholders if absent)."""
    import re as _re
    src = open(os.path.join(R, new_file)).read()
    for m in _re.finditer(r"from \.\.?(cpasync\.copy|typing) import \(([^)]*)\)", src):
        source = _cpcopy if m.group(1) == "cpasync.copy" else _cty
        for name in [n.strip().rstrip(",") for n in m.group(2).split() if n.strip().rstrip(",")]:
            if name in module.__dict__:
                continue
            module.__dict__[name] = getattr(source, name) if hasattr(source, name) else type(name, (), {})
            if not hasattr(source, name):
                print(f"[shim] placeholder type for {name}", flush=True)
if not hasattr(nvgpu, "make_tiled_tma_atom_C"):
    _prefill_names(nvh, "dsl_new_cute_nvgpu_helpers.py")
    _port(nvh, "dsl_new_cute_nvgpu_helpers.py", "make_tiled_tma_atom_C", export_to=nvgpu)
import cutlass.pipeline.sm90 as _sm90, cutlass.pipeline.helpers as _ph, inspect as _inspect
def _patch_advance(cls):
    orig = cls.__dict__["advance"]
    def advance(self, count=1, *, loc=None, ip=None):
        if not isinstance(count, int):
            raise NotImplementedError("shim: dynamic advance count")
        for _ in range(count):
            orig(self)
    advance.__name__ = "advance"
    cls.advance = advance
_n = 0
for _mod in (_sm90, _ph):
    for _name, _cls in vars(_mod).items():
        if _inspect.isclass(_cls) and "advance" in _cls.__dict__ and _cls.__module__ == _mod.__name__:
            try:
                if "count" in _inspect.signature(_cls.__dict__["advance"]).parameters:
                    continue
            except (TypeError, ValueError):
                pass
            _patch_advance(_cls); _n += 1
print(f"[shim] advance(count) added on {_n} pipeline classes", flush=True)
import cutlass.testing as _ct
_orig_benchmark = _ct.benchmark
def _benchmark_no_cupti(callable, **kw):
    if kw.get("use_cupti"):
        kw["use_cupti"] = False  # CUPTI activity API rejects this device with the available libcupti; CUDA-event path instead
        if not getattr(_benchmark_no_cupti, "_said", False):
            print("[shim] testing.benchmark: use_cupti=False (CUDA-event timing)", flush=True); _benchmark_no_cupti._said = True
    return _orig_benchmark(callable, **kw)
_ct.benchmark = _benchmark_no_cupti
sys.argv = [os.path.join(FMHA, "gqa.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
