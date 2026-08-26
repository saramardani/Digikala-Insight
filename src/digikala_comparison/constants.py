"""Column contracts for the pinned Digikala dataset."""

PRODUCT_REQUIRED_COLUMNS = (
    "id",
    "title_fa",
    "Rate",
    "Rate_cnt",
    "Category1",
    "Category2",
    "Brand",
    "Price",
    "Seller",
    "Is_Fake",
    "min_price_last_month",
    "sub_category",
)

COMMENT_REQUIRED_COLUMNS = (
    "id",
    "title",
    "body",
    "created_at",
    "rate",
    "recommendation_status",
    "is_buyer",
    "product_id",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes",
    "seller_title",
    "seller_code",
    "true_to_size_rate",
)

VALID_RECOMMENDATION_STATUSES = (
    "recommended",
    "not_recommended",
    "no_idea",
)
