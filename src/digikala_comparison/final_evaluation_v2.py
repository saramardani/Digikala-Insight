"""Foundation artifacts for the staged, reproducible final evaluation v2.

This module deliberately creates no evaluation examples and executes no model.
Later phases add frozen cases for each product capability to this contract.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Settings


V2_MANIFEST_NAME = "evaluation_manifest.json"
V2_CASE_SET_NAME = "evaluation_cases.json"
V2_HUMAN_TEMPLATE_NAME = "human_answer_quality_template.csv"
V2_README_NAME = "README.md"

_HUMAN_COLUMNS = (
    "case_id",
    "response_id",
    "component",
    "annotator_id",
    "status",
    "relevance_1_to_5",
    "clarity_1_to_5",
    "source_separation_1_to_5",
    "recommendation_usefulness_1_to_5",
    "uncertainty_appropriateness_1_to_5",
    "citation_usefulness_1_to_5",
    "semantic_citation_support_1_to_5",
    "reference_correctness",
    "failure_category",
    "comments",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinalEvaluationV2Case(_StrictModel):
    """One frozen input/reference pair for a final-system evaluation component."""

    case_id: str
    component: Literal["product_search", "review_qa", "comparison", "manager_analysis"]
    split: Literal["development", "final"]
    scenario: str
    user_input: str
    product_ids: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    reference: dict[str, Any]
    expected_status: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _validate_shape(self) -> "FinalEvaluationV2Case":
        if not self.case_id.strip() or not self.scenario.strip():
            raise ValueError("case_id and scenario must be non-empty")
        if not self.user_input.strip() and not (
            self.component == "product_search" and self.scenario == "empty_reference"
        ):
            raise ValueError("user_input must be non-empty except for product-search empty_reference")
        if len(self.product_ids) != len(set(self.product_ids)):
            raise ValueError("product_ids must be unique within an evaluation case")
        if self.component in {"review_qa", "comparison"} and not self.product_ids:
            raise ValueError(f"{self.component} cases require at least one product_id")
        if self.component == "comparison" and len(self.product_ids) < 2:
            raise ValueError("comparison cases require at least two product_ids")
        if not self.reference:
            raise ValueError("each frozen evaluation case requires an explicit reference")
        return self


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        item.update({"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return item


def build_evaluation_v2_manifest(settings: Settings) -> dict[str, Any]:
    """Capture reproducibility inputs without reading or writing API secrets."""

    selected_retriever = "unknown"
    selection_path = settings.paths.reranker_selection_report
    if selection_path is not None and selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_retriever = str(selection.get("selected_method", selected_retriever))

    return {
        "schema_version": settings.final_evaluation_v2.manifest_schema_version,
        "evaluation_version": settings.final_evaluation_v2.evaluation_version,
        "status": "foundation_initialized_pending_cases",
        "dataset": asdict(settings.dataset),
        "random_seed": settings.random_seed,
        "case_contract": {
            "schema_version": settings.final_evaluation_v2.case_schema_version,
            "components": ["product_search", "review_qa", "comparison", "manager_analysis"],
            "case_set_status": "empty_pending_staged_case_freeze",
        },
        "retrieval": {
            "selected_production_retriever": selected_retriever,
            "experiment_manifest": _artifact_fingerprint(settings.paths.retrieval_experiment_manifest),
        },
        "input_artifacts": {
            "canonical_products": _artifact_fingerprint(settings.paths.canonical_products),
            "product_statistics": _artifact_fingerprint(settings.paths.product_statistics),
            "retrieval_corpus": _artifact_fingerprint(settings.paths.retrieval_corpus),
        },
        "generation": {
            "provider": settings.generation.provider,
            "model": settings.generation.model,
            "prompt_version": settings.generation.prompt_version,
            "schema_version": settings.generation.schema_version,
            "api_key_environment_variable": settings.generation.api_key_environment_variable,
            "secret_value_persisted": False,
        },
        "grounding": {
            "validator_version": settings.grounding.validator_version,
            "unsupported_claim_action": settings.grounding.unsupported_claim_action,
            "human_rubric_version": settings.final_evaluation_v2.human_rubric_version,
        },
    }


def _write_json_if_absent(path: Path, value: object) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def initialize_final_evaluation_v2(settings: Settings, *, output_root: Path | None = None) -> dict[str, Path]:
    """Create idempotent v2 foundation artifacts; never mutate final_v1."""

    root = output_root or settings.paths.final_evaluation_v2_root
    if root is None:
        raise ValueError("final_evaluation_v2_root must be configured")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / V2_MANIFEST_NAME
    case_path = root / V2_CASE_SET_NAME
    template_path = root / V2_HUMAN_TEMPLATE_NAME
    readme_path = root / V2_README_NAME

    manifest = build_evaluation_v2_manifest(settings)
    _write_json_if_absent(manifest_path, manifest)
    _write_json_if_absent(
        case_path,
        {
            "schema_version": settings.final_evaluation_v2.case_schema_version,
            "case_set_status": "empty_pending_staged_case_freeze",
            "cases": [],
        },
    )
    if not template_path.exists():
        template_path.write_text(",".join(_HUMAN_COLUMNS) + "\n", encoding="utf-8")
    if not readme_path.exists():
        readme_path.write_text(
            "# Final system evaluation v2\n\n"
            "This directory is initialized but intentionally contains no evaluation cases yet. "
            "Cases are frozen separately by component in later phases; do not use this empty set to report quality.\n",
            encoding="utf-8",
        )
    return {"root": root, "manifest": manifest_path, "case_set": case_path, "human_template": template_path, "readme": readme_path}


def load_v2_cases(root: Path) -> tuple[dict[str, Any], list[FinalEvaluationV2Case]]:
    """Load and validate the frozen v2 case file."""

    payload = json.loads((root / V2_CASE_SET_NAME).read_text(encoding="utf-8"))
    return payload, [FinalEvaluationV2Case.model_validate(item) for item in payload["cases"]]


def append_frozen_component_cases(
    settings: Settings,
    *,
    root: Path,
    component: Literal["product_search", "review_qa", "comparison", "manager_analysis"],
    cases: list[FinalEvaluationV2Case],
) -> Path:
    """Append one component once; an existing component cannot be silently replaced."""

    payload, existing = load_v2_cases(root)
    if any(case.component == component for case in existing):
        raise ValueError(f"{component} cases are already frozen; create a new evaluation version to replace them")
    if not cases or any(case.component != component for case in cases):
        raise ValueError("a non-empty, single-component case collection is required")
    all_cases = [*existing, *cases]
    if len({case.case_id for case in all_cases}) != len(all_cases):
        raise ValueError("evaluation case_id values must be globally unique")
    payload["cases"] = [case.model_dump(mode="json") for case in all_cases]
    payload["case_set_status"] = "partially_frozen"
    payload["frozen_components"] = sorted({case.component for case in all_cases})
    payload["case_set_sha256"] = sha256(_stable_json(payload["cases"]).encode("utf-8")).hexdigest()
    (root / V2_CASE_SET_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest_path = root / V2_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "partially_frozen"
    manifest["case_contract"]["case_set_status"] = "partially_frozen"
    manifest["case_contract"]["frozen_components"] = payload["frozen_components"]
    manifest["case_contract"]["case_set_sha256"] = payload["case_set_sha256"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return root / V2_CASE_SET_NAME
