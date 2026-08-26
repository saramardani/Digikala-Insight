import polars as pl

from digikala_comparison.ingestion import clean_comments
from digikala_comparison.config import NormalizationSettings


def test_recommendation_status_keeps_unknown_values_auditable() -> None:
    comments = pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "product_id": ["p", "p", "p"],
            "title": ["t", "t", "t"],
            "body": ["b", "b", "b"],
            "is_buyer": ["True", "false", "possibly"],
            "recommendation_status": [" recommended ", "unknown_label", "nan"],
            "rate": ["4", "bad", "nan"],
            "likes": ["1", "2", "3"],
            "dislikes": ["0", "0", "0"],
            "advantages": ["nan", "nan", "nan"],
            "disadvantages": ["nan", "nan", "nan"],
        }
    ).lazy()
    settings = NormalizationSettings("NFC", True, True, True, True)

    cleaned = clean_comments(comments, settings).collect()

    assert cleaned[0, "recommendation_status_normalized"] == "recommended"
    assert cleaned[1, "recommendation_status_normalized"] is None
    assert cleaned[1, "recommendation_status_state"] == "unknown"
    assert cleaned[2, "recommendation_status_state"] == "missing"
    assert cleaned[0, "is_buyer_bool"] is True
    assert cleaned[1, "is_buyer_bool"] is False
    assert cleaned[2, "is_buyer_bool"] is None
