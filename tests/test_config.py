from pathlib import Path

from digikala_comparison.config import Settings


def test_default_paths_are_resolved_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_toml(root / "config" / "default.toml")

    assert settings.paths.raw_products == root / "data" / "raw" / "digikala-products.csv"
    assert settings.paths.quality_report == root / "data" / "reports" / "data_quality_report.json"
    assert settings.dataset.repository == "RadeAI/Digikala_comments_products"
    assert settings.dataset.revision == "89c3133b169c8d3793db8834f56f32fee33d9db0"
    assert settings.comparison.minimum_percentage_denominator == 20
    assert settings.comparison.practical_price_relative_difference == 0.05
    assert settings.generation.provider == "metis_openai_compatible"
    assert settings.generation.model == "gpt-4.1-mini"
    assert settings.generation.api_base_url == "https://api.metisai.ir/openai/v1"
    assert settings.generation.cost_estimation_available is False
    assert settings.paths.generation_cache_root == root / "data" / "generation" / "cache"
    assert settings.paths.generation_output_root == root / "data" / "generation" / "outputs"
    assert settings.grounding.unsupported_claim_action == "remove_unsupported"
    assert settings.paths.grounding_audit_root == root / "data" / "generation" / "grounding_audits"
    assert settings.final_evaluation.evaluation_version == "final-evaluation-v1"
    assert settings.paths.final_evaluation_root == root / "data" / "evaluations" / "final_v1"
