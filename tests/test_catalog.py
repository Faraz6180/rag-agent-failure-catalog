"""Smoke and behaviour tests. Offline only: no API keys, no network.

Two kinds of test:
  * every catalog script runs to completion, and every reproduce.py reports the failure;
  * the detectors and mitigations do what their READMEs claim on the bundled examples.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils import load_catalog_module, split_sentences, strip_citations, support_detail  # noqa: E402

FOLDERS = sorted(p.name for p in (ROOT / "catalog").iterdir() if p.is_dir() and p.name[:2].isdigit())


def _run(script: Path) -> str:
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("folder", FOLDERS)
def test_reproduce_reports_failure(folder: str) -> None:
    assert "FAILURE REPRODUCED" in _run(ROOT / "catalog" / folder / "reproduce.py")


@pytest.mark.parametrize("folder", FOLDERS)
@pytest.mark.parametrize("script", ["detect.py", "mitigate.py"])
def test_detect_and_mitigate_run(folder: str, script: str) -> None:
    _run(ROOT / "catalog" / folder / script)


@pytest.mark.parametrize("folder", FOLDERS)
def test_folder_has_expected_files(folder: str) -> None:
    for name in ["README.md", "reproduce.py", "detect.py", "mitigate.py"]:
        assert (ROOT / "catalog" / folder / name).exists()


# ------------------------------------------------------------------ utils
def test_support_ignores_chunk_labels() -> None:
    score, missing = support_detail("It takes 2 days.", "[2] Something else entirely.")
    assert "2" in missing and score < 1.0


def test_sentence_split_and_citation_strip() -> None:
    assert len(split_sentences("A is 99.9% up [1]. B is 5 [2].")) == 2
    assert strip_citations("Pro is $99 [1].") == "Pro is $99."


# ------------------------------------------------------------------ detectors and mitigations
def test_01_gate_blocks_ungrounded_and_passes_grounded() -> None:
    rep = load_catalog_module("01_ungrounded_answer", "reproduce")
    mit = load_catalog_module("01_ungrounded_answer", "mitigate")
    from common.retrieval_simulator import KeywordRetriever

    client = rep.get_client(rep.simulated_model)
    assert not mit.guarded_answer(client, rep.QUESTION, KeywordRetriever()).grounded
    assert mit.guarded_answer(client, "What is the refund policy for annual plans?", KeywordRetriever()).grounded
    assert load_catalog_module("01_ungrounded_answer", "detect").evaluate()["precision"] == 1.0


def test_02_detector_accuracy_and_repair() -> None:
    det = load_catalog_module("02_citation_drift", "detect")
    mit = load_catalog_module("02_citation_drift", "mitigate")
    rep = load_catalog_module("02_citation_drift", "reproduce")
    assert det.evaluate()["accuracy"] == 1.0
    repaired, dropped = mit.repair_citations(rep.simulated_model("", "", 0.0), rep.CHUNKS)
    assert "[3]" in repaired and len(dropped) == 1


def test_03_expansion_improves_answer_recall() -> None:
    rep = load_catalog_module("03_retrieval_surface_metrics", "reproduce")
    mit = load_catalog_module("03_retrieval_surface_metrics", "mitigate")
    from common.retrieval_simulator import KeywordRetriever

    r = KeywordRetriever()
    before = rep.summarize(rep.evaluate(r))["answer_recall@k"]
    after = rep.summarize(rep.evaluate(r, rewrite=mit.expand_query))["answer_recall@k"]
    assert before < after


def test_04_guard_stops_loop_and_ignores_healthy_run() -> None:
    rep = load_catalog_module("04_tool_calling_loops", "reproduce")
    det = load_catalog_module("04_tool_calling_loops", "detect")
    mit = load_catalog_module("04_tool_calling_loops", "mitigate")
    from common.agent_sim import run_agent

    assert det.detect_loop(rep.run_unguarded(verbose=False)).looping
    assert not det.detect_loop(det._healthy_trace()).looping
    t = run_agent(rep.TASK, rep.looping_policy, {"search_docs": rep.search_docs}, max_steps=12, step_hook=mit.LoopGuard(), verbose=False)
    assert t.stop_reason == "halted" and len(t.steps) < 12


def test_05_gate_forces_complete_answer() -> None:
    rep = load_catalog_module("05_premature_termination", "reproduce")
    det = load_catalog_module("05_premature_termination", "detect")
    mit = load_catalog_module("05_premature_termination", "mitigate")
    from common.agent_sim import run_agent

    t = run_agent(rep.TASK, rep.premature_policy, {"search_docs": rep.search_docs}, final_hook=mit.CompletionGate(rep.CHECKLIST), verbose=False)
    assert det.checklist_coverage(t.final_answer, rep.CHECKLIST).ratio == 1.0
    assert det.evaluate()["agreement"] == 1.0


def test_06_gate_routes_correctly() -> None:
    rep = load_catalog_module("06_overconfident_oos", "reproduce")
    mit = load_catalog_module("06_overconfident_oos", "mitigate")
    from common.retrieval_simulator import KeywordRetriever

    gate = mit.ScopeGate(KeywordRetriever())
    for item in rep.QUESTIONS:
        assert gate.decide(item["q"]).action == mit.EXPECTED[item["kind"]], item["q"]


def test_07_position_gap_and_mitigation() -> None:
    rep = load_catalog_module("07_context_pollution", "reproduce")
    det = load_catalog_module("07_context_pollution", "detect")
    mit = load_catalog_module("07_context_pollution", "mitigate")
    client = rep.get_client(rep.simulated_model)
    assert det.sensitivity(rep.position_sweep(client))["gap"] > 0.5
    assert rep.EXPECTED in rep.ask(client, mit.prepare_context(rep.QUESTION, rep.build_context(6)))


def test_08_detects_inconsistency() -> None:
    rep = load_catalog_module("08_inconsistent_behavior", "reproduce")
    det = load_catalog_module("08_inconsistent_behavior", "detect")
    client = rep.get_client(rep.make_stub())
    report = det.consistency_report(det.collect(lambda q, t: rep.ask(client, q, t), rep.PARAPHRASES), rep.EXPECTED_FACT)
    assert report.inconsistent


def test_09_typed_tool_surfaces_failure() -> None:
    rep = load_catalog_module("09_silent_tool_failure", "reproduce")
    det = load_catalog_module("09_silent_tool_failure", "detect")
    mit = load_catalog_module("09_silent_tool_failure", "mitigate")
    svc = rep.OrderService(down=True)
    assert rep.make_swallowing_tool(svc)(customer_id="C-1042") == "[]"
    assert det.classify_observation(mit.typed_tool(lambda customer_id: svc.lookup(customer_id))(customer_id="C-1042")) == "error"
    assert not det.canary(rep.make_swallowing_tool(svc), {"customer_id": "C-1042"})


def test_10_atom_detector_and_fallback() -> None:
    rep = load_catalog_module("10_numeric_hallucination", "reproduce")
    det = load_catalog_module("10_numeric_hallucination", "detect")
    mit = load_catalog_module("10_numeric_hallucination", "mitigate")
    assert det.evaluate()["recall"] == 1.0
    from common.llm_client import OfflineStub

    out = mit.verify_or_repair(OfflineStub(lambda s, u, t: rep.FAULTY), rep.QUESTION, rep.CHUNKS)
    assert out.path == "extractive_fallback"
    assert mit.verify_or_repair(OfflineStub(rep.simulated_model), rep.QUESTION, rep.CHUNKS).path == "regenerated"
