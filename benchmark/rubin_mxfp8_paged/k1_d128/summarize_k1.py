#!/usr/bin/env python3
"""Summarize K1 jsonl outputs: validation table (val.jsonl) or timing table (timing_32k.jsonl)."""

import json
import statistics
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def arm(r):
    if r["kernel"] == "product":
        return "product dense"
    return f"bench p{r['paged']} {r['corr']} cf{r['corrfast']} h{r['hoist']}" + ("" if r.get("prefolded_via", "params") == "params" else " (env)")


def val_table(rows):
    hdr = f"{'tag':22} {'arm':34} {'S':>5} {'mask':6} {'H':5} {'ramp':4} {'grow':4} {'rel_err':>8} {'max_abs':>9} {'fin':3} {'vs_dump':>18} {'resc default/always':>22}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        v = r.get("validation") or {}
        c = r.get("compare_o")
        cmp = "-" if c is None else ("bitwise" if c["bitwise"] else f"rel {c['rel_diff']:.2e}")
        rs = r.get("rescale_stats")
        rst = "-" if not rs else f"{rs['default']['rescaling_warp_steps']}/{rs['always']['rescaling_warp_steps']} of {rs['default']['live_warp_steps']}"
        print(
            f"{r.get('tag',''):22} {arm(r):34} {r['S']:>5} {r['mask']:6} {str(r['h_q'])+'/'+str(r['h_kv']):5} {r.get('kv_ramp',0):4} {r.get('kv_growth',1.0):4} "
            f"{v.get('rel_err', float('nan')):8.4f} {v.get('max_abs_err', float('nan')):9.2e} {'ok' if v.get('finite', True) else 'NaN':3} {cmp:>18} {rst:>22}"
        )


def timing_table(rows):
    by = {}
    for r in rows:
        by.setdefault(arm(r), []).append(r)
    print(f"{'arm':34} {'n':>2} {'median us':>10} {'min':>8} {'max':>8} {'TF/s':>7}  clocks per burst (sm MHz, max, temp, W)")
    for a, rs in by.items():
        ts = [r["time_us_median"] for r in rs]
        med = statistics.median(ts)
        S, B, HQ, D = rs[0]["S"], rs[0]["B"], rs[0]["h_q"], rs[0]["d"]
        flops = 4.0 * B * HQ * S * S * D * (0.5 * (1 + 1.0 / S) if rs[0]["mask"] == "causal" else 1.0)
        clocks = "; ".join(c for r in rs for c in r.get("clocks_per_burst", []))
        print(f"{a:34} {len(ts):>2} {med:10.1f} {min(ts):8.1f} {max(ts):8.1f} {flops / (med * 1e-6) / 1e12:7.0f}  {clocks[:160]}")


if __name__ == "__main__":
    rows = load(sys.argv[1])
    if rows and "time_us_median" in rows[0]:
        timing_table(rows)
    else:
        val_table(rows)
