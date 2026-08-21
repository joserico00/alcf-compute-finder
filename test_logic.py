"""
Logic tests for the ALCF Compute Finder that need NO streamlit install.

They execute the real app.py end to end with a faithful mock of the streamlit
API: widgets return their defaults unless a test overrides them by label, and
everything the app renders is recorded so tests can assert on it. This covers
the ranking engine, hard-requirement misses, score breakdowns, the compare
table, filters, and every tab's code path - everything except pixels.

Run:  python3 test_logic.py        (only needs pandas)

The sibling test_app.py does the same scenarios through streamlit's official
AppTest harness - use that where streamlit is installed.
"""

import contextlib
import sys
import types


def make_streamlit(overrides):
    """A stand-in streamlit module. Widgets return their default value unless
    `overrides` maps their label to something else."""
    st = types.ModuleType("streamlit")
    recorded = {"warning": [], "subheader": [], "success": [], "info": []}
    st._recorded = recorded

    def _noop(*a, **kw):
        return None

    def selectbox(label, options, index=0, **kw):
        options = list(options)
        value = overrides.get(label, options[index])
        assert value in options, f"override {value!r} not a valid option for {label!r}"
        return value

    def slider(label, min_value=0, max_value=100, value=0, **kw):
        return overrides.get(label, value)

    def checkbox(label, value=False, **kw):
        return overrides.get(label, value)

    def multiselect(label, options, default=None, **kw):
        return overrides.get(label, list(default) if default else [])

    def text_input(label, value="", **kw):
        return overrides.get(label, value)

    def tabs(labels):
        return [contextlib.nullcontext() for _ in labels]

    def columns(spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [contextlib.nullcontext() for _ in range(n)]

    def expander(label, **kw):
        return contextlib.nullcontext()

    def progress(v, **kw):
        assert 0.0 <= v <= 1.0, f"progress out of range: {v}"

    def record(name):
        def f(msg=None, *a, **kw):
            recorded[name].append(msg)
        return f

    st.selectbox = selectbox
    st.slider = slider
    st.checkbox = checkbox
    st.multiselect = multiselect
    st.text_input = text_input
    st.tabs = tabs
    st.columns = columns
    st.expander = expander
    st.progress = progress
    st.sidebar = contextlib.nullcontext()
    st.session_state = {}   # plain dict: supports .get() and [] like the real one
    st.warning = record("warning")
    st.subheader = record("subheader")
    st.success = record("success")
    st.info = record("info")
    for name in ("set_page_config", "markdown", "write", "caption", "code",
                 "download_button", "dataframe", "divider"):
        setattr(st, name, _noop)
    return st


def run_app(**overrides):
    """Execute the real app.py under the mock and return its namespace."""
    st = make_streamlit(overrides)
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st
    try:
        ns = {"__name__": "app_under_test", "__file__": "app.py"}
        with open("app.py", encoding="utf-8") as fh:
            exec(compile(fh.read(), "app.py", "exec"), ns)
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)
    ns["_recorded"] = st._recorded
    return ns


def best(ns):
    return ns["top"]["sys"]["name"]


# --- scenarios --------------------------------------------------------------

TASK = "What are you doing?"
SCALE = "How much do you need at once?"
HW = "Hardware you need"
MIN_ACCEL = "Accelerators per node, minimum"
MIN_ACCEL_MEM = "Accelerator memory per node (GB), minimum"
NOTEBOOKS = "I need JupyterHub notebooks"


def test_smoke_default():
    ns = run_app()
    assert best(ns)
    assert ns["clean"], "default settings should leave clean matches"
    # score must equal the sum of its own breakdown, for every system
    for r in ns["results"]:
        assert r["score"] == sum(p for p, _ in r["breakdown"]), r["sys"]["name"]


def test_llm_api_recommends_inference_service():
    ns = run_app(**{TASK: "Use an LLM through an API (no setup)"})
    assert best(ns) == "ALCF Inference Service", best(ns)


def test_single_gpu_training_recommends_sophia():
    ns = run_app(**{TASK: "Train or fine-tune a model",
                    SCALE: "Part of one accelerator",
                    HW: "NVIDIA (CUDA)"})
    assert best(ns) == "Sophia", best(ns)


def test_huge_simulation_recommends_aurora():
    ns = run_app(**{TASK: "Large-scale simulation (MPI / CFD / MD / QCD)",
                    SCALE: "Hundreds to thousands of nodes"})
    assert best(ns) == "Aurora", best(ns)


def test_cpu_only_data_work_recommends_crux():
    ns = run_app(**{TASK: "Data analysis, preprocessing, workflows",
                    HW: "CPU only"})
    assert best(ns) == "Crux", best(ns)


def test_novel_hardware_recommends_a_testbed_system():
    ns = run_app(**{TASK: "Benchmark or port code to novel AI hardware",
                    HW: "Non-GPU AI accelerator"})
    assert best(ns) in {"Cerebras CS-3", "SambaNova DataScale (SN30)",
                        "Graphcore Bow Pod64", "GroqRack"}, best(ns)


def test_notebooks_requirement_picks_a_jupyter_system():
    ns = run_app(**{TASK: "Learn HPC, teach a class, run notebooks",
                    NOTEBOOKS: True})
    assert best(ns) in {"Polaris", "Sophia", "ALCF Inference Service"}, best(ns)
    # every clean match must actually satisfy the notebook requirement
    for r in ns["clean"]:
        s = r["sys"]
        assert s["jupyter"] or s["accel_class"] == "service", s["name"]


def test_impossible_requirement_warns():
    # 800 GB accel memory per node exceeds every published spec (Aurora tops
    # out at 768). Alone, that still leaves the Inference Service clean - by
    # design, per-node minimums do not apply to a managed service. Adding an
    # accelerators-per-node minimum rules the service out too, so nothing at
    # all fits and the app must warn.
    ns = run_app(**{MIN_ACCEL_MEM: 800, MIN_ACCEL: 1})
    assert not ns["clean"]
    assert len(ns["_recorded"]["warning"]) >= 1, "expected the 'nothing fits' warning"
    assert best(ns)  # a closest match is still offered


def test_service_is_exempt_from_per_node_minimums():
    # Memory minimum alone: the managed Inference Service must stay clean.
    ns = run_app(**{MIN_ACCEL_MEM: 800})
    assert [r["sys"]["key"] for r in ns["clean"]] == ["inference"]


def test_unknown_specs_never_satisfy_minimums():
    ns = run_app(**{TASK: "Train or fine-tune a model", MIN_ACCEL: 8})
    # Sophia, GroqRack and SambaNova SN30 all publish 8 accelerators per node
    assert best(ns) in {"Sophia", "GroqRack", "SambaNova DataScale (SN30)"}, best(ns)
    for r in ns["clean"]:
        assert r["sys"]["accel_per_node"] >= 8, r["sys"]["name"]
    # Graphcore has no meaningful per-node IPU count (SLURM assigns them), so
    # it must register as a miss, never a silent pass
    gc = next(r for r in ns["results"] if r["sys"]["key"] == "graphcore")
    assert any("not published" in m for m in gc["misses"]), gc["misses"]


def test_unpublished_node_counts_miss_scale_requirements():
    # Minerva and Janus publish no node structure; with any node-scale need
    # they must carry a 'node count not published' miss (the fixed bug).
    ns = run_app(**{SCALE: "One node"})
    for key in ("minerva", "janus"):
        r = next(r for r in ns["results"] if r["sys"]["key"] == key)
        assert "node count not published" in r["misses"], (key, r["misses"])


def test_miss_wording_says_only():
    ns = run_app(**{MIN_ACCEL: 8})
    polaris = next(r for r in ns["results"] if r["sys"]["key"] == "polaris")
    assert any(m.startswith("only 4 ") for m in polaris["misses"]), polaris["misses"]


def test_misses_always_rank_below_clean():
    ns = run_app(**{TASK: "Run inference at scale"})
    seen_missed = False
    for r in ns["results"]:
        if r["misses"]:
            seen_missed = True
        else:
            assert not seen_missed, "a clean system ranked below a missed one"


def test_presets_are_valid_and_fill_widgets():
    ns = run_app()
    # every preset value must be a real option of the widget it fills
    for name, values in ns["PRESETS"].items():
        assert values["task_sel"] in ns["TASKS"], name
        assert values["scale_sel"] in ns["SCALES"], name
        assert values["accel_sel"] in ns["ACCEL_CHOICES"], name
    # firing the callback must copy the preset into widget state and reset
    # the hard-requirement sliders
    st = ns["st"]
    st.session_state["min_accel_sl"] = 8          # stale leftover to be reset
    st.session_state["preset_sel"] = "Brand new: learn HPC with notebooks"
    ns["apply_preset"]()
    assert st.session_state["task_sel"] == "Learn HPC, teach a class, run notebooks"
    assert st.session_state["nb_chk"] is True
    assert st.session_state["min_accel_sl"] == 0
    # the no-op choice must change nothing
    st.session_state["preset_sel"] = ns["CUSTOM_SCENARIO"]
    st.session_state["task_sel"] = "sentinel"
    ns["apply_preset"]()
    assert st.session_state["task_sel"] == "sentinel"


def test_all_task_scale_combinations_render():
    ns0 = run_app()
    tasks = list(ns0["TASKS"].keys())
    scales = list(ns0["SCALES"].keys())
    for task in tasks:
        for scale in scales:
            ns = run_app(**{TASK: task, SCALE: scale})
            assert best(ns), f"no recommendation for {task!r} @ {scale!r}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"FAIL  {name}: {exc!r}")
    print(f"\n{'ALL TESTS PASSED' if not failures else str(failures) + ' FAILURE(S)'}")
    sys.exit(1 if failures else 0)
