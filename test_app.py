"""
Tests for the ALCF Compute Finder (app.py).

Run either way:

    pip install streamlit pandas pytest
    pytest test_app.py -q          # or:  python test_app.py

Uses Streamlit's official AppTest harness: it executes the real app script
headlessly, drives the sidebar widgets, and inspects the rendered output.

Sidebar widget order (indices used below):
    selectbox[0] = task     selectbox[1] = scale      selectbox[2] = hardware
    slider[0]    = min accelerators/node
    slider[1]    = min accelerator memory (GB)
    slider[2]    = min CPU cores
    slider[3]    = min CPU memory (GB)
    checkbox[0]  = notebooks   checkbox[1] = available now   checkbox[2] = hide misses
"""

from streamlit.testing.v1 import AppTest


def run_app(task=None, scale=None, hardware=None,
            min_accel=None, min_accel_mem=None, notebooks=None):
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception, f"App raised on first run: {at.exception}"

    sb = at.sidebar
    if task is not None:
        sb.selectbox[0].select(task)
    if scale is not None:
        sb.selectbox[1].select(scale)
    if hardware is not None:
        sb.selectbox[2].select(hardware)
    if min_accel is not None:
        sb.slider[0].set_value(min_accel)
    if min_accel_mem is not None:
        sb.slider[1].set_value(min_accel_mem)
    if notebooks is not None:
        sb.checkbox[0].set_value(notebooks)

    at.run()
    assert not at.exception, f"App raised after widget changes: {at.exception}"
    return at


def best_match(at):
    """The best-match system is the first (only) st.subheader on the page."""
    return at.subheader[0].value


def test_smoke_default():
    at = run_app()
    assert best_match(at)  # something is always recommended


def test_llm_api_recommends_inference_service():
    at = run_app(task="Use an LLM through an API (no setup)")
    assert best_match(at) == "ALCF Inference Service"


def test_single_gpu_training_recommends_sophia():
    at = run_app(task="Train or fine-tune a model",
                 scale="Part of one accelerator",
                 hardware="NVIDIA (CUDA)")
    assert best_match(at) == "Sophia"


def test_huge_simulation_recommends_aurora():
    at = run_app(task="Large-scale simulation (MPI / CFD / MD / QCD)",
                 scale="Hundreds to thousands of nodes")
    assert best_match(at) == "Aurora"


def test_cpu_only_data_work_recommends_crux():
    at = run_app(task="Data analysis, preprocessing, workflows",
                 hardware="CPU only")
    assert best_match(at) == "Crux"


def test_novel_hardware_recommends_a_testbed_system():
    at = run_app(task="Benchmark or port code to novel AI hardware",
                 hardware="Non-GPU AI accelerator")
    assert best_match(at) in {
        "Cerebras CS-3", "SambaNova DataScale (SN30)",
        "Graphcore Bow Pod64", "GroqRack",
    }


def test_notebooks_requirement_picks_a_jupyter_system():
    at = run_app(task="Learn HPC, teach a class, run notebooks", notebooks=True)
    assert best_match(at) in {"Polaris", "Sophia", "ALCF Inference Service"}


def test_impossible_requirement_shows_warning():
    # 800 GB accelerator memory per node exceeds every published spec (Aurora
    # tops out at 768). Alone, that still leaves the Inference Service clean -
    # by design, per-node minimums do not apply to a managed service. Adding
    # an accelerators-per-node minimum rules the service out too, so nothing
    # fits and the app must warn.
    at = run_app(min_accel=1, min_accel_mem=800)
    assert len(at.warning) >= 1


def test_unknown_specs_never_satisfy_minimums():
    # Systems with unpublished per-node accel counts (e.g. SambaNova) must not
    # win when a per-node accelerator minimum is set. With min 8/node the only
    # clean GPU answers are Sophia (8) and GroqRack (8).
    at = run_app(task="Train or fine-tune a model", min_accel=8)
    assert best_match(at) in {"Sophia", "GroqRack"}


def test_all_task_scale_combinations_render():
    """Brute-force smoke: every task x scale combination must render clean."""
    tasks = [
        "Use an LLM through an API (no setup)",
        "Train or fine-tune a model",
        "Run inference at scale",
        "Large-scale simulation (MPI / CFD / MD / QCD)",
        "Data analysis, preprocessing, workflows",
        "Learn HPC, teach a class, run notebooks",
        "Benchmark or port code to novel AI hardware",
        "Visualization",
    ]
    scales = [
        "Part of one accelerator", "One node", "2-10 nodes",
        "10-100 nodes", "Hundreds to thousands of nodes",
    ]
    for task in tasks:
        for scale in scales:
            at = run_app(task=task, scale=scale)
            assert best_match(at), f"No recommendation for {task!r} @ {scale!r}"


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failures else 0)
