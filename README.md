# ALCF Compute Finder

A guide for new (and not-so-new) Argonne Leadership Computing Facility users:
describe the work you need done, and it ranks ALCF's systems against it —
Aurora, Polaris, Sophia, Crux, the AI Testbed machines, the Inference Service,
and the announced Tara / Minerva / Janus — then hands you the exact commands
for your first job.

## What it does

- **Ranks systems for your workload.** Pick a task, a scale, and any hard
  hardware requirements in the sidebar. Systems that miss a hard requirement
  are demoted below every system that meets them all, no matter their score.
- **Shows its work.** Every recommendation comes with a point-by-point score
  breakdown — the ranking is never a black box.
- **Quick-start presets.** One click fills in the whole sidebar for common
  scenarios ("Fine-tune a model on NVIDIA GPUs", "Brand new: learn HPC", ...).
- **First-job scripts.** A ready-to-submit PBS (or SLURM) script per system,
  with a line-by-line plain-English explainer.
- **Compare mode.** Any 2–3 systems side by side, including their match
  scores for your current requirements.
- **New here? tab.** An HPC glossary in plain English and an interactive
  first-week onboarding checklist.

A design rule worth knowing: **unknown specs never satisfy a requirement.**
If a system doesn't publish a number (e.g. per-node details for Minerva),
it registers as a miss rather than silently passing.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+ and Streamlit 1.46+ (for `width="stretch"`).

## Tests

Two suites cover the same scenarios:

```bash
python test_logic.py        # needs only pandas - runs app.py under a mock streamlit
pytest test_app.py -q       # needs streamlit installed - uses the official AppTest harness
```

`test_logic.py` is the one to run in constrained environments; it executes
the real `app.py` end to end (ranking engine, misses, breakdowns, presets,
every tab) with the UI layer stubbed.

## Updating the hardware specs

All system data lives in the `SYSTEMS` list at the top of `app.py`, with the
source URLs recorded per system in `sources`. When hardware or queue policy
changes:

1. Check the pages listed in the file header (docs.alcf.anl.gov + alcf.anl.gov).
2. Edit the numbers in `SYSTEMS`.
3. Bump `LAST_VERIFIED` (it is displayed in the UI so users know how stale
   the data may be).
4. Run `python test_logic.py` — the tests assert ranking behavior, so a data
   change that breaks an expectation will surface immediately.

Specs were last verified **August 21, 2026**.

## Deploying

### Option A - Streamlit Community Cloud (recommended)

The standard way to host a Streamlit app for free from a GitHub repo:

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app** → pick this repo and branch → main file `app.py` → Deploy.
4. You get a public `https://<name>.streamlit.app` URL. Pushes to the branch
   redeploy automatically.

### Option B - GitHub Pages (serverless, via stlite)

GitHub Pages only serves static files, so a normal Streamlit app can't run
there — but the included `index.html` runs the app **in the browser** with
[stlite](https://github.com/whitphx/stlite) (Streamlit on WebAssembly):

1. Push the repo (must include `index.html` and `app.py`) to GitHub.
2. Repo **Settings → Pages** → Source: *Deploy from a branch* →
   Branch: `main`, folder `/ (root)` → Save.
3. After a minute, the app is live at
   `https://<username>.github.io/<repo-name>/`.

Trade-off: the first visit downloads the Python runtime (roughly 20–60 s,
cached afterwards), and everything runs client-side. For a snappier
experience use Option A; for zero-infrastructure hosting use Option B.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The whole app: system data, ranking engine, UI |
| `index.html` | Serverless GitHub Pages build (stlite/WebAssembly) |
| `requirements.txt` | `streamlit`, `pandas` |
| `test_logic.py` | Dependency-light test suite (mocked streamlit) |
| `test_app.py` | Same scenarios via Streamlit's official AppTest harness |

## Disclaimer

Hardware, queues, and policies change. Confirm with the linked ALCF
documentation or support@alcf.anl.gov before making capacity or proposal
decisions.
