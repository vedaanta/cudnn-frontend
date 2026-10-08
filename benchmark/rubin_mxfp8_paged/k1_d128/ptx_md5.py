#!/usr/bin/env python3
"""Render the sm_107a PTX of the d128 MXFP8 kernel (product file or the bench one-off under the current BENCH_* env)
at the HALF + prefolded dense TemplateParams and print its md5 + statement count.  One arm per process (the DSL
reads CUTE_DSL_* at its first import; the bench kernel reads BENCH_* at module exec).

  ptx_md5.py <dump_dir> product|bench [paged 0|1]
"""

import glob
import hashlib
import os
import sys

dump, arm = sys.argv[1], sys.argv[2]
paged = len(sys.argv) > 3 and sys.argv[3] == "1"
os.makedirs(dump, exist_ok=True)
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
os.environ["CUTE_DSL_CACHE_DIR"] = os.path.join(dump, "dsl_cache")  # a JIT-cache hit would dump nothing
os.makedirs(os.environ["CUTE_DSL_CACHE_DIR"], exist_ok=True)
if arm == "bench":
    os.environ.setdefault("BENCH_PAGED64", "1" if paged else "0")

from cudnn.sdpa.fwd import api_dsl  # noqa: E402
from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module  # noqa: E402
from cudnn.sdpa.fwd.config_sm100 import TemplateParams  # noqa: E402

if arm == "bench":
    api_dsl._SM107_MXFP8_KERNEL_FILES[(128, 128)] = "sm107/prefill_d128_mxfp8_bench.py"
kw = dict(dtype_qkv=0, dtype_o=2, cta_mma=2, softmax_f16=True, softmax_scale_prefolded=True)
if paged:
    kw.update(paged_kv=True, page_size=128, seq_kv_lens_present=True)
params = TemplateParams(**kw)
mod = _load_sm100_kernel_module((128, 128), params, fp8=True, pertensor=False, rubin=True)
levers = {k: getattr(mod, k, None) for k in ("PAGED_KV", "BENCH_CORR", "BENCH_CORRFAST", "BENCH_HOIST", "SOFTMAX_F16", "SCALE_PREFOLDED", "_FUSED_SHIFT_CVT")}
mod.compile_prepared(d_qk=128, d_v=128, has_lse=False, lse_kind="dense", has_amax=False)
ptxs = sorted(glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True), key=os.path.getmtime)
if not ptxs:
    print("FAIL no ptx dumped into", dump, os.listdir(dump))
    sys.exit(3)
with open(ptxs[-1], "rb") as f:
    data = f.read()
lines = [ln for ln in data.decode().splitlines() if ln.strip() and not ln.strip().startswith("//")]
# the DSL stamps the function name with the module's load ordinal; compare the body with names normalized
import re  # noqa: E402

norm = re.sub(rb"cudnn_frontend_[A-Za-z0-9_]+|_templates_[A-Za-z0-9_]+|sdpa_fwd_sm107_mxfp8_d128[A-Za-z0-9_]*", b"NAME", data)
print(
    f"ARM {arm} paged={int(paged)} levers={levers} ptx_files={len(ptxs)} statements={len(lines)} md5_raw={hashlib.md5(data).hexdigest()} md5_norm={hashlib.md5(norm).hexdigest()} file={ptxs[-1]}"
)
