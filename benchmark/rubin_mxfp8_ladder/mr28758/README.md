# Running DKG MR 28758 (`gqa.py`, 2xfp8 GQA/MLA prefill) on the FROST venv

`run_gqa.py` runs the MR's `gqa.py` on the installed internal CuTe DSL (0.3.0, 2026-07-28) by shimming the APIs the MR-era DSL
added: `ifence` (no-op, or the real `.pragma "next knob FenceInterference"` with `GQA_IFENCE=1`), `cutlass.memory.get_smem_capacity_in_bytes`
(`GQA_SMEM_BYTES`, 327680 on Rubin), `utils.sm100.make_smem_layout_c` and `cute.nvgpu.make_tiled_tma_atom_C` (ported verbatim from the
MR-era DSL source files `blackwell_helpers_new.py` / `dsl_new_cute_nvgpu_helpers.py`, NOT committed), `PipelineProducer/Consumer.advance(count)`,
and `testing.benchmark(use_cupti=False)` (CUPTI's activity API rejects the device with the available libcupti).  The MR sources
(`gqa.py`, `mask.py`, `gqa_decode.py`, `helpers/`) are fetched from gitlab-master at the MR head and are NOT committed here.
Layout expected by `run_gqa.py`: `<dir>/CuTeDSL/{helpers,blackwell/kernel/attention/fmha}` next to the two DSL source files.
`smoke.sh` / `compare.sh` / `compare2.sh` are the board drivers (board_py.sh with `PYTHONPATH_TAIL` for the cupti-python side install).
Results: `../results/P6_gqa_mr28758_*`.
