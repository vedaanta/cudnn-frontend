#!/usr/bin/env python3
"""SASS helpers for the cc 10.7 MXFP8 bench: carve the cubin out of a compiled-plan ``kernel.o``,
disassemble it with the toolkit's ``cuobjdump``/``nvdisasm``, and histogram the opcodes per pipe class.

  sass_tools.py carve <kernel.o> <out.cubin>              # raw cubin embedded in the host relocatable (.lrodata)
  sass_tools.py sass <cubin> <out.sass> [--tk <toolkit>]  # cuobjdump -sass (needs nvdisasm_internal on PATH for sm_107a)
  sass_tools.py hist <file.sass> [--json out.json]        # opcode histogram by pipe class (static, whole kernel)
  sass_tools.py ncu-source <ncu_source.csv> [--json out]  # executed-instruction histogram from `ncu --import --page source --csv --print-source sass`

The compiled-plan cache (``XDG_CACHE_HOME/cudnn/compiled_plans/v2/<key>/<entry>/kernel.o``) stores the kernel as an
x86-64 relocatable whose ``.lrodata`` holds the sm_107a cubin verbatim (not fatbin-wrapped); ``carve`` finds every
embedded ELF with e_machine == 190 (EM_CUDA) and writes the largest one.
"""
import argparse
import collections
import csv
import json
import os
import re
import struct
import subprocess
import sys

EM_CUDA = 190

# opcode -> pipe class (coarse; what the unit-SOL model needs).  Anything else -> "other".
PIPE_CLASS = [
    (re.compile(r"^(UTCQMMA|UTCMMA|UTCHMMA|UTCBMMA|HMMA|IMMA|QMMA|OMMA|UTCCP|UTCBAR|UTCRESCALE|UTCALLOC|UTCDEALLOC|UTCRELINQ|UTCCOMMIT|UTCCTL)"), "tensor"),
    (re.compile(r"^MUFU"), "xu"),
    (re.compile(r"^(FFMA2|FFMA|FMUL|FADD|FMNMX|FSET|FSEL|FCHK|FFMA2|FMUL2|FADD2|FMNMX2|HFMA2|HMUL2|HADD2|HMNMX2|HFMA|HMUL|HADD|FHADD2|FHFMA2|FSETP|HSETP2|FRND|FSWZADD)"), "fma"),
    (re.compile(r"^(F2FP|F2F|I2F|F2I|I2I|I2FP|F2IP|FRND)"), "alu_cvt"),
    (re.compile(r"^(IADD3|IMAD|IMNMX|ISETP|LOP3|SHF|SHL|SHR|LEA|SEL|PRMT|MOV|IABS|PLOP3|P2R|R2P|POPC|FLO|BREV|VOTE|VOTEU|LOP|IADD|ISET|BMSK|REDUX|R2UR|S2R|S2UR|CS2R|NOP|PSETP|LDC|ULDC|UMOV|UIADD3|UIMAD|ULOP3|USHF|USEL|ULEA|UISETP|UPLOP3|UP2UR|UFLO|UPOPC|UBMSK|UPRMT|USETMAXREG|R2B|B2R|VIADD|VIMNMX|VABSDIFF|IDP)"), "alu_int"),
    (re.compile(r"^(LDTM|STTM|TCGEN05|LDTMB|STTMB)"), "tmem"),
    (re.compile(r"^(LDS|STS|LDSM|STSM|ATOMS|LDGSTS|LD\.|ST\.|LDG|STG|LDL|STL|RED|ATOM|ATOMG|MEMBAR|FENCE|ERRBAR|CCTL|UTMALDG|UTMASTG|UTMACCTL|UTMAPF|UTMACMDFLUSH|UTMAREDG|LD$|ST$)"), "lsu_mem"),
    (re.compile(r"^(SYNCS|USYNCS|BAR|UBAR|WARPSYNC|DEPBAR|NANOSLEEP|BPT|EXIT|BRA|BRX|JMP|JMX|RET|CALL|BSYNC|BSSY|BREAK|YIELD|WARPGROUP|UCGABAR|CGAERRBAR|ELECT|UELECT|VOTE)"), "ctrl_sync"),
]


def classify(op):
    base = op.split(".")[0]
    for rx, cls in PIPE_CLASS:
        if rx.match(base):
            return cls
    return "other"


FATBIN_MAGIC = b"\x50\xed\x55\xba"  # u32 0xBA55ED50 little-endian, the fatbin container header


def carve(path, out):
    """Write the device code embedded in a compiled-plan kernel.o: the whole fatbin container when its header
    is present (cuobjdump reads fatbins directly), else the largest raw EM_CUDA ELF (size from its program AND
    section headers -- cubins place the program headers last)."""
    b = open(path, "rb").read()
    i = b.find(FATBIN_MAGIC)
    if i >= 0:
        _magic, _ver, hsz, total = struct.unpack_from("<IHHQ", b, i)
        size = hsz + total
        open(out, "wb").write(b[i : i + size])
        print(f"carved fatbin: offset {i} size {size} -> {out}")
        return out
    best = None
    i = b.find(b"\x7fELF")
    while i >= 0:
        if len(b) >= i + 64:
            (em,) = struct.unpack_from("<H", b, i + 18)
            if em == EM_CUDA:
                (phoff, shoff) = struct.unpack_from("<QQ", b, i + 32)
                phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", b, i + 54)
                end = max(shoff + shentsize * shnum, phoff + phentsize * phnum)
                for s in range(shnum):
                    o = i + shoff + s * shentsize
                    if o + 64 > len(b):
                        break
                    (sh_type,) = struct.unpack_from("<I", b, o + 4)
                    sh_off, sh_size = struct.unpack_from("<QQ", b, o + 24)
                    if sh_type != 8:  # SHT_NOBITS has no file bytes
                        end = max(end, sh_off + sh_size)
                size = min(end, len(b) - i)
                if best is None or size > best[1]:
                    best = (i, size)
        i = b.find(b"\x7fELF", i + 1)
    if best is None:
        raise SystemExit(f"no fatbin / EM_CUDA ELF found inside {path}")
    off, size = best
    open(out, "wb").write(b[off : off + size])
    print(f"carved cubin: offset {off} size {size} -> {out}")
    return out


def sass(cubin, out, tk=None):
    tk = tk or os.environ.get("CUDA_TK") or "/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest"
    env = dict(os.environ)
    env["PATH"] = f"{tk}/bin:" + env.get("PATH", "")  # cuobjdump dispatches to nvdisasm_internal for sm_107a
    r = subprocess.run([f"{tk}/bin/cuobjdump", "-sass", cubin], capture_output=True, text=True, env=env)
    if r.returncode != 0 or "/*0" not in r.stdout:
        r2 = subprocess.run([f"{tk}/bin/nvdisasm", "-c", cubin], capture_output=True, text=True, env=env)
        if r2.returncode != 0:
            raise SystemExit(f"cuobjdump: {r.stderr[:300]}\nnvdisasm: {r2.stderr[:300]}")
        r = r2
    open(out, "w").write(r.stdout)
    n = sum(1 for l in r.stdout.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", l))
    print(f"wrote {out}: {n} instructions")
    return out


_SASS_LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def hist(path, as_json=None):
    ops = collections.Counter()
    for line in open(path, errors="ignore"):
        m = _SASS_LINE.match(line)
        if m:
            ops[m.group(2)] += 1
    by_class = collections.Counter()
    for op, n in ops.items():
        by_class[classify(op)] += n
    total = sum(ops.values())
    print(f"{total} instructions (static)")
    for cls, n in by_class.most_common():
        print(f"  {cls:10s} {n:7d}  {100.0 * n / max(total, 1):5.1f}%")
    print("top opcodes:", ", ".join(f"{o} {n}" for o, n in ops.most_common(25)))
    rec = dict(static_total=total, by_class=dict(by_class), opcodes=dict(ops.most_common()))
    if as_json:
        json.dump(rec, open(as_json, "w"), indent=1)
    return rec


def ncu_source(path, as_json=None):
    """Executed warp-instructions per opcode from ncu's source page (SASS view, --set full SourceCounters)."""
    rows = list(csv.reader(open(path, errors="ignore")))
    hdr_i = next((i for i, r in enumerate(rows) if r and any(c.strip() in ("Source", "# Address", "Address") for c in r)), None)
    if hdr_i is None:
        raise SystemExit("no SASS source table header found")
    hdr = rows[hdr_i]
    idx = {h.strip(): i for i, h in enumerate(hdr)}
    src_col = idx.get("Source")
    exe_col = next((i for h, i in idx.items() if h.lower().replace(" ", "_") in ("instructions_executed", "warp_instructions_executed", "inst_executed", "# instructions executed")), None)
    if exe_col is None:
        exe_col = next((i for h, i in idx.items() if "Executed" in h and "Instructions" in h), None)
    if src_col is None or exe_col is None:
        raise SystemExit(f"columns not found; header = {hdr[:12]}")
    ops = collections.Counter()
    for r in rows[hdr_i + 1 :]:
        if len(r) <= max(src_col, exe_col):
            continue
        m = re.match(r"\s*(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", r[src_col])
        if not m:
            continue
        try:
            n = float(r[exe_col].replace(",", ""))
        except ValueError:
            continue
        ops[m.group(1)] += n
    by_class = collections.Counter()
    for op, n in ops.items():
        by_class[classify(op)] += n
    total = sum(ops.values())
    print(f"{total:.0f} warp-instructions executed (ncu)")
    for cls, n in by_class.most_common():
        print(f"  {cls:10s} {n:14.0f}  {100.0 * n / max(total, 1):5.1f}%")
    rec = dict(executed_total=total, by_class=dict(by_class), opcodes=dict(ops.most_common()))
    if as_json:
        json.dump(rec, open(as_json, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("carve"); p.add_argument("kernel_o"); p.add_argument("out")
    p = sub.add_parser("sass"); p.add_argument("cubin"); p.add_argument("out"); p.add_argument("--tk", default=None)
    p = sub.add_parser("hist"); p.add_argument("sass"); p.add_argument("--json", default=None)
    p = sub.add_parser("ncu-source"); p.add_argument("csv"); p.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "carve":
        carve(a.kernel_o, a.out)
    elif a.cmd == "sass":
        sass(a.cubin, a.out, a.tk)
    elif a.cmd == "hist":
        hist(a.sass, a.json)
    else:
        ncu_source(a.csv, a.json)


if __name__ == "__main__":
    main()
