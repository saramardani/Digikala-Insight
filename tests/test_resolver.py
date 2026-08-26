from __future__ import annotations

import polars as pl

from digikala_comparison.config import ResolutionSettings
from digikala_comparison.product_identity import normalize_product_text
from digikala_comparison.resolver import ProductResolver


def _resolver() -> ProductResolver:
    rows = [
        ("1", "سامسونگ A55 5G 128GB", "Samsung", "mobile"),
        ("2", "سامسونگ A55 4G 128GB", "Samsung", "mobile"),
        ("3", "سامسونگ A35 5G 128GB", "Samsung", "mobile"),
        ("4", "Redmi Note 13 Pro", "Xiaomi", "mobile"),
        ("5", "Redmi Note 13 Pro+", "Xiaomi", "mobile"),
        ("6", "کالای مشترک", "A", "one"),
        ("7", "کالای مشترک", "B", "two"),
    ]
    records = []
    for product_id, title, brand, category in rows:
        records.append(
            {
                "product_id": product_id,
                "title_fa": title,
                "normalized_title": normalize_product_text(title),
                "normalized_brand": normalize_product_text(brand),
                "Brand": brand,
                "Category1": category,
                "Category2": None,
                "sub_category": category,
                "canonicalization_status": "unique_source_row",
                "total_review_count": 3,
                "buyer_review_count": 2,
                "recommendation_known_count": 2,
            }
        )
    return ProductResolver(pl.DataFrame(records), ResolutionSettings())


def test_exact_id_and_normalized_title_resolution() -> None:
    resolver = _resolver()
    assert resolver.resolve(1).status == "exact"
    result = resolver.resolve("  سامسونگ  A55 / 5G 128 GB ")
    assert result.status == "exact"
    assert result.selected_product_id == "1"


def test_brand_aware_fuzzy_and_model_protection() -> None:
    resolver = _resolver()
    fuzzy = resolver.resolve({"title": "سامسون a55 5g 128gb", "brand": "Samsung"})
    assert fuzzy.selected_product_id == "1"
    assert resolver.resolve("samsung a25 5g 128gb").status == "not_found"
    assert resolver.resolve("redmi note 13 pro+").selected_product_id == "5"
    assert resolver.resolve("redmi note 13 pro").selected_product_id == "4"


def test_ambiguous_not_found_and_multi_resolution() -> None:
    resolver = _resolver()
    assert resolver.resolve("کالای مشترک").status == "ambiguous"
    assert resolver.resolve("product that does not exist").status == "not_found"
    results = resolver.resolve_many([1, "redmi note 13 pro+"])
    assert [item.selected_product_id for item in results] == ["1", "5"]
