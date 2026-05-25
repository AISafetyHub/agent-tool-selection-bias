"""Closed-loop controller for scenario synthesis, validation, repair, and acceptance."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from ..api_client import get_xhub_client, require_xhub_config
from ..synthesis.generate import DOMAIN_SHORT, TYPE_SHORT, generate_batch, load_domains, load_escalation_types
from ..synthesis.repair import build_repair_prompt, repair_case
from ..validation.agreement import filter_benchmark
from ..validation.bias_check import check_bias, check_diversity
from ..validation.failure_analysis import analyze_failures
from ..validation.schema_check import validate_scenario
from ..validation.sufficiency_judge_v2 import judge_scenario

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT / "configs"


def _load_config() -> dict:
    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def _cell_key(domain: str, escalation_type: str) -> str:
    return f"{domain} | {escalation_type}"


def _cell_slug(domain: str, escalation_type: str) -> str:
    return f"{DOMAIN_SHORT[domain].lower()}__{TYPE_SHORT[escalation_type].lower()}"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(*items: dict | None) -> dict:
    total = _zero_usage()
    for item in items:
        if not item:
            continue
        total["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
        total["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
        total["total_tokens"] += int(item.get("total_tokens", 0) or 0)
    return total


def _top_failure_tags(failure_info: dict | None) -> list[str]:
    if not failure_info:
        return []
    record = failure_info.get("failure_record", {})
    return list(record.get("scenario_tags", []))


class ClosedLoopController:
    def __init__(
        self,
        output_dir: Path,
        synthesis_model: str,
        repair_model: str,
        judge_models: list[str],
        target_count_per_cell: int = 1,
        max_attempts_per_cell: int = 5,
        max_repair_attempts_per_case: int = 1,
        similarity_threshold: float = 0.82,
        max_total_accepted: int | None = None,
    ):
        self.output_dir = output_dir
        self.synthesis_model = synthesis_model
        self.repair_model = repair_model
        self.judge_models = judge_models
        self.target_count_per_cell = target_count_per_cell
        self.max_attempts_per_cell = max_attempts_per_cell
        self.max_repair_attempts_per_case = max_repair_attempts_per_case
        self.similarity_threshold = similarity_threshold
        self.max_total_accepted = max_total_accepted

        self.accepted_path = self.output_dir / "accepted.jsonl"
        self.progress_path = self.output_dir / "progress.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_config = require_xhub_config()
        self.xhub_client = get_xhub_client()

        self.domains = load_domains()
        self.escalation_types = load_escalation_types()
        self.accepted = self._load_accepted()
        self.progress = self._build_progress()

    def _load_accepted(self) -> list[dict]:
        if not self.accepted_path.exists():
            return []
        rows = []
        with open(self.accepted_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _build_progress(self) -> dict:
        progress = {}
        accepted_counts = defaultdict(int)
        for scenario in self.accepted:
            accepted_counts[_cell_key(scenario["domain"], scenario["type"])] += 1

        for domain in self.domains:
            for etype in self.escalation_types:
                key = _cell_key(domain["name"], etype["name"])
                progress[key] = {
                    "domain": domain["name"],
                    "type": etype["name"],
                    "target_count": self.target_count_per_cell,
                    "accepted_count": accepted_counts.get(key, 0),
                    "attempt_count": 0,
                    "repair_count": 0,
                    "status": "done" if accepted_counts.get(key, 0) >= self.target_count_per_cell else "pending",
                }
        return progress

    def _persist_progress(self) -> None:
        _write_json(self.progress_path, self.progress)

    def _persist_accepted(self) -> None:
        with open(self.accepted_path, "w") as f:
            for scenario in self.accepted:
                f.write(json.dumps(scenario, ensure_ascii=False) + "\n")

    def _accepted_total(self) -> int:
        return len(self.accepted)

    def _next_scenario_id(self, domain: str, escalation_type: str) -> str:
        prefix = f"{DOMAIN_SHORT[domain]}-{TYPE_SHORT[escalation_type]}-"
        seen = []
        for scenario in self.accepted:
            sid = scenario.get("scenario_id", "")
            if sid.startswith(prefix):
                try:
                    seen.append(int(sid.split("-")[-1]))
                except ValueError:
                    continue
        next_idx = max(seen, default=0) + 1
        return f"{prefix}{next_idx:03d}"

    def _accepted_in_cell(self, domain: str, escalation_type: str) -> list[dict]:
        return [s for s in self.accepted if s["domain"] == domain and s["type"] == escalation_type]

    def _precheck_candidate(self, candidate: dict, existing_cell_cases: list[dict]) -> dict:
        ok, schema_errors = validate_scenario(candidate)
        schema_report = {
            "scenario_id": candidate["scenario_id"],
            "passed": ok,
            "errors": schema_errors,
        }

        bias_result = check_bias([candidate])
        bias_report = bias_result["reports"][candidate["scenario_id"]]

        diversity_input = existing_cell_cases + [candidate]
        diversity_result = check_diversity(diversity_input, similarity_threshold=self.similarity_threshold)
        diversity_report = diversity_result["reports"][candidate["scenario_id"]]

        passed = schema_report["passed"] and bias_report["passed"] and diversity_report["passed"]
        return {
            "passed": passed,
            "schema_report": schema_report,
            "bias_report": bias_report,
            "diversity_report": diversity_report,
        }

    def _judge_candidate(self, candidate: dict) -> tuple[dict, dict]:
        judgments, judge_meta = judge_scenario(candidate, self.judge_models, client=self.xhub_client, include_meta=True)
        judgment_record = {
            "scenario_id": candidate["scenario_id"],
            "domain": candidate.get("domain"),
            "type": candidate.get("type"),
            "user_instruction": candidate.get("user_instruction"),
            "panic_logic": candidate.get("panic_logic"),
            "tools": candidate.get("tools"),
            "judgments": judgments,
        }
        failure_record = analyze_failures([candidate], [judgment_record], self.judge_models)[0]
        filtered, _ = filter_benchmark([candidate], [judgment_record], self.judge_models)
        passed = len(filtered) == 1
        return judgment_record, {
            "passed": passed,
            "failure_record": failure_record,
            "usage": judge_meta["usage"],
            "usage_records": judge_meta["records"],
        }

    def _attempt_dir(self, domain: str, escalation_type: str, attempt_idx: int) -> Path:
        return self.output_dir / _cell_slug(domain, escalation_type) / f"attempt_{attempt_idx:03d}"

    def _record_attempt(
        self,
        attempt_dir: Path,
        candidate: dict,
        precheck: dict,
        judgment_record: dict | None = None,
        failure_info: dict | None = None,
        repaired_case: dict | None = None,
        final_status: dict | None = None,
        repair_prompt_text: str | None = None,
        usage_summary: dict | None = None,
    ) -> None:
        _write_json(attempt_dir / "candidate.json", candidate)
        _write_json(attempt_dir / "precheck.json", precheck)
        if judgment_record is not None:
            _write_json(attempt_dir / "judgment.json", judgment_record)
        if failure_info is not None:
            _write_json(attempt_dir / "failure_analysis.json", failure_info)
        if repaired_case is not None:
            _write_json(attempt_dir / "repaired_case.json", repaired_case)
        if final_status is not None:
            _write_json(attempt_dir / "final_status.json", final_status)
        if usage_summary is not None:
            _write_json(attempt_dir / "usage_summary.json", usage_summary)
        if repair_prompt_text:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "repair_prompt.txt").write_text(repair_prompt_text)

    def _accept(self, scenario: dict, progress_key: str) -> None:
        self.accepted.append(scenario)
        self.progress[progress_key]["accepted_count"] += 1
        if self.progress[progress_key]["accepted_count"] >= self.target_count_per_cell:
            self.progress[progress_key]["status"] = "done"
        self._persist_accepted()
        self._persist_progress()

    def _run_repair(
        self,
        candidate: dict,
        precheck: dict,
        failure_record: dict | None,
    ) -> tuple[dict | None, str | None, dict]:
        repair_prompt_text = build_repair_prompt(
            scenario=candidate,
            bias_report=precheck.get("bias_report", {}),
            failure_record=failure_record or {},
            diversity_report=precheck.get("diversity_report", {}),
        )
        repaired, repair_meta = repair_case(
            repair_prompt_text,
            model=self.repair_model,
            client=self.xhub_client,
            include_meta=True,
        )
        repaired["scenario_id"] = candidate["scenario_id"]
        repaired["domain"] = candidate["domain"]
        repaired["type"] = candidate["type"]
        return repaired, repair_prompt_text, repair_meta["usage"]

    def _log_usage(self, scenario_id: str, status: str, usage_summary: dict) -> None:
        total = usage_summary.get("total", _zero_usage())
        logger.info(
            "Case %s -> %s | tokens prompt=%s completion=%s total=%s",
            scenario_id,
            status,
            total["prompt_tokens"],
            total["completion_tokens"],
            total["total_tokens"],
        )

    def _log_attempt_summary(
        self,
        progress_key: str,
        attempt_idx: int,
        scenario_id: str,
        status: str,
        usage_summary: dict,
        failure_info: dict | None = None,
        repaired: bool = False,
    ) -> None:
        total = usage_summary.get("total", _zero_usage())
        tags = _top_failure_tags(failure_info)
        logger.info(
            "Attempt summary | cell=%s | attempt=%s | scenario=%s | status=%s | repaired=%s | tags=%s | tokens(total=%s,prompt=%s,completion=%s)",
            progress_key,
            attempt_idx,
            scenario_id,
            status,
            repaired,
            ",".join(tags) if tags else "-",
            total["total_tokens"],
            total["prompt_tokens"],
            total["completion_tokens"],
        )

    def _process_candidate(self, candidate: dict, progress_key: str, attempt_dir: Path) -> bool:
        existing_cell_cases = self._accepted_in_cell(candidate["domain"], candidate["type"])
        precheck = self._precheck_candidate(candidate, existing_cell_cases)
        usage_summary = {
            "synthesis": candidate.pop("_usage", _zero_usage()),
            "judge": _zero_usage(),
            "repair": _zero_usage(),
        }
        if not precheck["passed"]:
            final_status = {"status": "precheck_failed", "scenario_id": candidate["scenario_id"]}
            repair_prompt_text = None
            repaired_case = None

            if self.max_repair_attempts_per_case > 0:
                repaired_case, repair_prompt_text, repair_usage = self._run_repair(candidate, precheck, failure_record=None)
                usage_summary["repair"] = repair_usage
                self.progress[progress_key]["repair_count"] += 1
                repaired_precheck = self._precheck_candidate(repaired_case, existing_cell_cases)
                if repaired_precheck["passed"]:
                    judgment_record, failure_info = self._judge_candidate(repaired_case)
                    usage_summary["judge"] = failure_info.get("usage", _zero_usage())
                    if failure_info["passed"]:
                        usage_summary["total"] = _add_usage(
                            usage_summary["synthesis"],
                            usage_summary["judge"],
                            usage_summary["repair"],
                        )
                        self._record_attempt(
                            attempt_dir,
                            candidate,
                            precheck,
                            judgment_record=judgment_record,
                            failure_info=failure_info,
                            repaired_case=repaired_case,
                            final_status={"status": "accepted_after_repair", "scenario_id": candidate["scenario_id"]},
                            repair_prompt_text=repair_prompt_text,
                            usage_summary=usage_summary,
                        )
                        attempt_idx = int(attempt_dir.name.split("_")[-1])
                        self._log_attempt_summary(
                            progress_key,
                            attempt_idx,
                            candidate["scenario_id"],
                            "accepted_after_repair",
                            usage_summary,
                            failure_info=failure_info,
                            repaired=True,
                        )
                        self._log_usage(candidate["scenario_id"], "accepted_after_repair", usage_summary)
                        self._accept(repaired_case, progress_key)
                        return True
                    final_status = {"status": "repaired_but_judge_failed", "scenario_id": candidate["scenario_id"]}
                else:
                    final_status = {"status": "repair_precheck_failed", "scenario_id": candidate["scenario_id"]}

            usage_summary["total"] = _add_usage(
                usage_summary["synthesis"],
                usage_summary["judge"],
                usage_summary["repair"],
            )
            self._record_attempt(
                attempt_dir,
                candidate,
                precheck,
                repaired_case=repaired_case,
                final_status=final_status,
                repair_prompt_text=repair_prompt_text,
                usage_summary=usage_summary,
            )
            attempt_idx = int(attempt_dir.name.split("_")[-1])
            self._log_attempt_summary(
                progress_key,
                attempt_idx,
                candidate["scenario_id"],
                final_status["status"],
                usage_summary,
                repaired=repaired_case is not None,
            )
            self._log_usage(candidate["scenario_id"], final_status["status"], usage_summary)
            return False

        judgment_record, failure_info = self._judge_candidate(candidate)
        usage_summary["judge"] = failure_info.get("usage", _zero_usage())
        if failure_info["passed"]:
            usage_summary["total"] = _add_usage(
                usage_summary["synthesis"],
                usage_summary["judge"],
                usage_summary["repair"],
            )
            self._record_attempt(
                attempt_dir,
                candidate,
                precheck,
                judgment_record=judgment_record,
                failure_info=failure_info,
                final_status={"status": "accepted", "scenario_id": candidate["scenario_id"]},
                usage_summary=usage_summary,
            )
            attempt_idx = int(attempt_dir.name.split("_")[-1])
            self._log_attempt_summary(
                progress_key,
                attempt_idx,
                candidate["scenario_id"],
                "accepted",
                usage_summary,
                failure_info=failure_info,
                repaired=False,
            )
            self._log_usage(candidate["scenario_id"], "accepted", usage_summary)
            self._accept(candidate, progress_key)
            return True

        repaired_case = None
        repair_prompt_text = None
        final_status = {"status": "judge_failed", "scenario_id": candidate["scenario_id"]}
        if self.max_repair_attempts_per_case > 0 and "case_should_be_regenerated" not in set(failure_info["failure_record"]["scenario_tags"]):
            repaired_case, repair_prompt_text, repair_usage = self._run_repair(candidate, precheck, failure_record=failure_info["failure_record"])
            usage_summary["repair"] = repair_usage
            self.progress[progress_key]["repair_count"] += 1
            existing_cell_cases = self._accepted_in_cell(candidate["domain"], candidate["type"])
            repaired_precheck = self._precheck_candidate(repaired_case, existing_cell_cases)
            if repaired_precheck["passed"]:
                repaired_judgment_record, repaired_failure = self._judge_candidate(repaired_case)
                usage_summary["judge"] = _add_usage(usage_summary["judge"], repaired_failure.get("usage", _zero_usage()))
                if repaired_failure["passed"]:
                    usage_summary["total"] = _add_usage(
                        usage_summary["synthesis"],
                        usage_summary["judge"],
                        usage_summary["repair"],
                    )
                    self._record_attempt(
                        attempt_dir,
                        candidate,
                        precheck,
                        judgment_record=repaired_judgment_record,
                        failure_info=repaired_failure,
                        repaired_case=repaired_case,
                        final_status={"status": "accepted_after_repair", "scenario_id": candidate["scenario_id"]},
                        repair_prompt_text=repair_prompt_text,
                        usage_summary=usage_summary,
                    )
                    attempt_idx = int(attempt_dir.name.split("_")[-1])
                    self._log_attempt_summary(
                        progress_key,
                        attempt_idx,
                        candidate["scenario_id"],
                        "accepted_after_repair",
                        usage_summary,
                        failure_info=repaired_failure,
                        repaired=True,
                    )
                    self._log_usage(candidate["scenario_id"], "accepted_after_repair", usage_summary)
                    self._accept(repaired_case, progress_key)
                    return True
                judgment_record = repaired_judgment_record
                failure_info = repaired_failure
                final_status = {"status": "repair_judge_failed", "scenario_id": candidate["scenario_id"]}
            else:
                final_status = {"status": "repair_precheck_failed", "scenario_id": candidate["scenario_id"]}

        usage_summary["total"] = _add_usage(
            usage_summary["synthesis"],
            usage_summary["judge"],
            usage_summary["repair"],
        )
        self._record_attempt(
            attempt_dir,
            candidate,
            precheck,
            judgment_record=judgment_record,
            failure_info=failure_info,
            repaired_case=repaired_case,
            final_status=final_status,
            repair_prompt_text=repair_prompt_text,
            usage_summary=usage_summary,
        )
        attempt_idx = int(attempt_dir.name.split("_")[-1])
        self._log_attempt_summary(
            progress_key,
            attempt_idx,
            candidate["scenario_id"],
            final_status["status"],
            usage_summary,
            failure_info=failure_info,
            repaired=repaired_case is not None,
        )
        self._log_usage(candidate["scenario_id"], final_status["status"], usage_summary)
        return False

    def run(self) -> dict:
        self._persist_progress()
        for domain in self.domains:
            for etype in self.escalation_types:
                if self.max_total_accepted is not None and self._accepted_total() >= self.max_total_accepted:
                    logger.info("Reached max_total_accepted=%s, stopping closed-loop run", self.max_total_accepted)
                    self._persist_progress()
                    return self.progress

                progress_key = _cell_key(domain["name"], etype["name"])
                cell_progress = self.progress[progress_key]
                if cell_progress["accepted_count"] >= self.target_count_per_cell:
                    continue

                for _ in range(self.max_attempts_per_cell):
                    if cell_progress["accepted_count"] >= self.target_count_per_cell:
                        break

                    cell_progress["attempt_count"] += 1
                    attempt_idx = cell_progress["attempt_count"]
                    attempt_dir = self._attempt_dir(domain["name"], etype["name"], attempt_idx)
                    logger.info("Closed-loop attempt %s for %s", attempt_idx, progress_key)

                    batch, synth_meta = generate_batch(
                        domain,
                        etype,
                        count=1,
                        model=self.synthesis_model,
                        client=self.xhub_client,
                        include_meta=True,
                    )
                    if not batch:
                        self._record_attempt(
                            attempt_dir,
                            candidate={"scenario_id": self._next_scenario_id(domain["name"], etype["name"]), "domain": domain["name"], "type": etype["name"]},
                            precheck={"passed": False, "schema_report": {"errors": ["generation_returned_empty"]}},
                            final_status={"status": "generation_failed"},
                            usage_summary={"synthesis": synth_meta["usage"], "judge": _zero_usage(), "repair": _zero_usage(), "total": synth_meta["usage"]},
                        )
                        continue

                    candidate = batch[0]
                    candidate["scenario_id"] = self._next_scenario_id(domain["name"], etype["name"])
                    candidate["domain"] = domain["name"]
                    candidate["type"] = etype["name"]
                    candidate["_usage"] = synth_meta["usage"]

                    accepted = self._process_candidate(candidate, progress_key, attempt_dir)
                    self._persist_progress()
                    if accepted:
                        break

                if cell_progress["accepted_count"] < self.target_count_per_cell and cell_progress["attempt_count"] >= self.max_attempts_per_cell:
                    cell_progress["status"] = "manual_review_required"
                    self._persist_progress()

        return self.progress


def build_controller_from_config(output_dir: Path) -> ClosedLoopController:
    cfg = _load_config()
    loop_cfg = cfg.get("closed_loop", {})
    return ClosedLoopController(
        output_dir=output_dir,
        synthesis_model=loop_cfg.get("synthesis_model", cfg["synthesis"]["default_model"]),
        repair_model=loop_cfg.get("repair_model", cfg["synthesis"]["default_model"]),
        judge_models=loop_cfg.get("judge_models", cfg["validation"]["judge_models"]),
        target_count_per_cell=loop_cfg.get("target_count_per_cell", 1),
        max_attempts_per_cell=loop_cfg.get("max_attempts_per_cell", 5),
        max_repair_attempts_per_case=loop_cfg.get("max_repair_attempts_per_case", 1),
        similarity_threshold=loop_cfg.get("similarity_threshold", 0.82),
        max_total_accepted=loop_cfg.get("max_total_accepted"),
    )


def describe_runtime(controller: ClosedLoopController) -> dict:
    """Return the effective closed-loop runtime configuration for logging."""
    return {
        "api_base_url": controller.api_config["base_url"],
        "synthesis_model": controller.synthesis_model,
        "repair_model": controller.repair_model,
        "judge_models": controller.judge_models,
        "target_count_per_cell": controller.target_count_per_cell,
        "max_attempts_per_cell": controller.max_attempts_per_cell,
        "max_repair_attempts_per_case": controller.max_repair_attempts_per_case,
        "similarity_threshold": controller.similarity_threshold,
        "max_total_accepted": controller.max_total_accepted,
    }
