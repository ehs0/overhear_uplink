# AGENT.md

Working notes for anyone writing code in this repository, human or agent.

`README.md` explains **what** the codec does and how to run it. This file records
**how we work on it** — the conventions to follow and the reasons behind them.
When a new rule is agreed, add it here rather than leaving it in a chat log.

## What this code is for

A research baseline for one question:

> Can a robot exploit previously overheard uplink transmissions from other
> robots as side information to reduce its own uplink transmission cost while
> maintaining reconstruction quality?

Everything in `src/overhear_uplink/` exists to answer that, so changes should be
judged against it. Two consequences worth keeping in mind:

- **The comparison is the product, not the codec.** A change that improves the
  overhearing model must be applied to the no-overhear baseline too, or the
  rate-distortion comparison stops being an ablation. `use_overhear` is the only
  thing allowed to differ between the two arms.
- **The encoder and the decoder must stay synchronised.** Both the transmitting
  robot and the server compute the same prediction from the same previously
  reconstructed images. Anything the encoder conditions on must be derivable by
  the decoder, or the scheme is not implementable over a real uplink.

Current stage: source coding on synthetic views. Later stages — real multi-view
data, pose conditioning, imperfect overhearing, a wireless channel — are listed
in the research plan.

## Repository map

| Path | Role |
|---|---|
| `config.py` | `ModelConfig` dataclass; the single source of architecture truth, stored into every checkpoint |
| `layers.py` | Analysis/synthesis transforms, hyper transforms, `OverhearContext` cross-attention |
| `entropy.py` | Quantisation, Gaussian/factorised likelihoods, bit counting |
| `model.py` | `OverhearUplinkCodec` — the sequential codec; `RobotPacket` — one robot's transmission |
| `losses.py` | `RateDistortionLoss`; also computes the per-robot metrics |
| `engine.py` | Train and eval loops over a loader |
| `data.py` | `MultiRobotFolderDataset` (real scenes), `SyntheticMultiRobotDataset` (pipeline checks) |
| `runtime.py` | Device selection, dataset/loader construction, checkpoint config |
| `train.py` / `evaluate.py` / `compare.py` / `demo.py` | Entry points, run as `python -m overhear_uplink.<name>` |
| `configs/` | JSON experiment definitions — tracked, and the reproducible half of every run |
| `results/` | Measurements the study reports — tracked on purpose |

## Conventions

### `.gitignore`

**Group related entries together, and comment each group with what it matches
and why it is safe to ignore.**

A bare list of patterns tells the next reader nothing. The question that
actually comes up is not "is `runs/` ignored" but "why is `results/` tracked
when everything else a run produces is not" — and only a comment can answer it.

```gitignore
# Training run outputs
#   runs/   one directory per experiment: resolved_config.json + checkpoints
#   *.pt    checkpoints written anywhere outside runs/
#   logs/   per-epoch stdout from overhear_uplink.train
# Tens of MB per run, and fully reproducible from the tracked configs/ via
# launch_gpu_sweep.sh. The recipe is versioned; the weights it produces are not.
runs/
*.pt
logs/
```

When adding a pattern, put it in the group it belongs to and extend that group's
comment. Start a new group only for a genuinely new category. After editing,
confirm the change did what you intended:

```bash
git status --short --ignored=matching | grep '^!!' | sort
```

Compare that list before and after — for a pure reorganisation it must be
identical.

### Python

These are what the existing code already does; match it rather than introducing
a second style.

- `from __future__ import annotations` at the top of every module (`__init__.py`
  is the exception — it carries no annotations).
- Type hints on every function signature, modern syntax (`list[str]`,
  `Tensor | None`).
- Docstrings on public classes and on functions whose contract is not obvious
  from the signature. Module-level docstrings are used only for entry points,
  where they become the `--help` text.
- Validate arguments where they enter and raise `ValueError` with a message that
  names the offending value. See `ModelConfig.__post_init__` and
  `OverhearUplinkCodec._validate_image`.
- No new runtime dependencies without discussion — the environment is currently
  `torch` and `pillow` only.

### Experiments

- An experiment is defined by a JSON file in `configs/`, never by editing
  defaults in code. `train.py` copies the resolved config into the output
  directory so a checkpoint always carries the settings that produced it.
- Both arms of a comparison get identical seed, schedule, data and epoch budget.
- Report rate as what it is: the entropy-model estimate, not a coded bitstream.
  Both arms are measured the same way, so comparisons hold, but absolute
  bitrates are optimistic.

### Commits

- Conventional-commit subject (`feat:`, `docs:`, `fix:`), imperative mood.
- The body says why the change was needed, not just what changed. For an
  experiment, record the headline number and the caveats that qualify it.

## Adding to this file

Keep entries short and give the reason, not just the rule — a rule without its
reason gets dropped the first time it is inconvenient. Record conventions that
are actually followed; if the code and this file disagree, one of them is a bug.
