from __future__ import annotations

import csv
from pathlib import Path

import pytest

from digikala_comparison.constants import COMMENT_REQUIRED_COLUMNS, PRODUCT_REQUIRED_COLUMNS


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def source_files(tmp_path: Path) -> tuple[Path, Path]:
    products = write_csv(
        tmp_path / "digikala-products.csv",
        PRODUCT_REQUIRED_COLUMNS,
        [
            {
                "id": "001",
                "title_fa": "کالای تستی",
                "Rate": "90",
                "Rate_cnt": "2",
                "Category1": "آزمون",
                "Category2": "",
                "Brand": "متفرقه",
                "Price": "1000",
                "Seller": "فروشنده",
                "Is_Fake": "False",
                "min_price_last_month": "0",
                "sub_category": "test",
            }
        ],
    )
    comments = write_csv(
        tmp_path / "digikala-comments.csv",
        COMMENT_REQUIRED_COLUMNS,
        [
            {
                "id": "007",
                "title": "عنوان",
                "body": "متن تست",
                "created_at": "1 فروردین 1400",
                "rate": "0",
                "recommendation_status": "recommended",
                "is_buyer": "True",
                "product_id": "001",
                "advantages": "nan",
                "disadvantages": "nan",
                "likes": "0",
                "dislikes": "0",
                "seller_title": "فروشنده",
                "seller_code": "T1",
                "true_to_size_rate": "nan",
            }
        ],
    )
    return products, comments
