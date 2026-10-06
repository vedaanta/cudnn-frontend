"""How often does a correction warp rescale?  Emulate the kernel's lazy-threshold rule on the real logits.
Per row: running max m (scaled log2 units), update when cur_max - m > THRESH (first tile always); a warp
(32 consecutive rows of a 128-row sub-tile) rescales on a step if ANY of its rows updates on a step other
than its first live step (first PV has accumulate=False, no rescale).  Also reports the per-half (64-token)
variant used by rungs G/H."""
import math, os, sys
sys.path.insert(0, os.getcwd())
import torch
import bench_ladder as b

S = int(sys.argv[1]); causal = sys.argv[2] == "causal"; THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
case = b.build_case("F", S, causal)
q, k, v, scale = case["ref_inputs"]           # dequantized fp32 (B, H, S, D)
scale_log2 = scale * math.log2(math.e)
Bn, H, _, D = q.shape
kh = k.shape[1]; grp = H // kh

def run(tile):
    n_tiles = S // tile
    steps_total = 0; resc = 0; warps_live = 0
    for h in range(H):
        qh = q[0, h]; khh = k[0, h // grp]
        m = torch.full((S,), float("-inf"), device=q.device)
        first = torch.ones((S,), dtype=torch.bool, device=q.device)
        for t in range(n_tiles):
            s_blk = (qh @ khh[t * tile:(t + 1) * tile].T) * scale_log2          # (S, tile)
            if causal:
                rows = torch.arange(S, device=q.device)[:, None]; cols = (t * tile + torch.arange(tile, device=q.device))[None, :]
                s_blk = s_blk.masked_fill(cols > rows, float("-inf"))
            cur = s_blk.amax(dim=1)
            live = torch.isfinite(cur)
            upd = live & (first | ((cur - m) > THRESH))
            m = torch.where(upd, cur, m)
            # a rescale is an update on a row that already has live content
            resc_rows = upd & ~first & live
            first = first & ~live
            # warps: 32 consecutive rows; count warps with any live row this step, and those with any rescale row
            live_w = live.view(-1, 32).any(dim=1); resc_w = resc_rows.view(-1, 32).any(dim=1)
            warps_live += int(live_w.sum()); resc += int(resc_w.sum())
    return resc, warps_live

for tile in (128, 64):
    r, n = run(tile)
    print(f"S={S} {'causal' if causal else 'none'} thresh={THRESH} tile={tile}: warp-steps with a rescale {r} / live warp-steps {n} = {100*r/max(n,1):.2f}%")
