# پیشنهاد معماری و پایپلاین برای بخش ۳: مقایسه‌ی محصولات

برای **بخش ۳: مقایسه‌ی محصولات**، پیشنهاد من این است که اصلاً از ابتدا سراغ Agent پیچیده یا Fine-tuning نروید. مسئله‌ی اصلی شما یک **Evidence-grounded Comparison Pipeline** است: داده‌ی ساخت‌یافته‌ی محصول + بازیابی نظرات مرتبط + استخراج نقاط قوت/ضعف + مقایسه‌ی عددی + LLM برای ساخت پاسخ نهایی.

طبق صورت پروژه، مقایسه باید بین دو یا چند محصول انجام شود و معیارهایی مثل قیمت، مشخصات، رضایت کاربران، نقاط قوت/ضعف پرتکرار و شواهد نظرات را پوشش دهد. مهم‌تر اینکه باید مرز بین «داده‌ی مستقیم»، «شواهد کاربران» و «استنتاج مدل» مشخص باشد.

همچنین پاسخ مبتنی بر نظر باید به داده‌ی واقعی متکی باشد و شناسه‌ی نظرات استفاده‌شده قابل ارائه باشد.

---

## معماری پیشنهادی

من Pipeline را به این شکل می‌سازم:

```text
User Query
   ↓
Query Parser
   ↓
Product Resolver
   ↓
┌───────────────────────────────┐
│ Structured Product Data      │
│ price / brand / rate / ...   │
└───────────────────────────────┘
   +
┌───────────────────────────────┐
│ Review Retrieval             │
│ Hybrid Search                │
│ Dense + Sparse               │
│        ↓                     │
│ Reranker                     │
└───────────────────────────────┘
   ↓
Evidence Extraction
   ↓
Aspect Aggregation
   ↓
Comparison Engine
   ↓
LLM Answer Generator
   ↓
Grounding Validator
   ↓
Final Comparison
+ review IDs
+ evidence
+ confidence
```

یعنی LLM را **تصمیم‌گیر اصلی روی داده‌ی خام قرار نمی‌دهیم**. Python ابتدا شواهد و آمار را تولید می‌کند؛ LLM صرفاً آن‌ها را به یک پاسخ طبیعی فارسی تبدیل می‌کند.

قاعده‌ی تصمیم‌گیری در داده‌های متناقض نیز باید صریح باشد: وجود چند نظر مثبت یا منفی به‌تنهایی نتیجه را تعیین نمی‌کند. برای هر معیار، ابتدا تعداد و **درصد** شواهد مثبت، منفی و خنثی محاسبه می‌شود؛ سپس نتیجه با ذکر حجم نمونه بیان می‌شود. درصد بالا با تعداد بسیار کم نباید به یک ادعای قطعی تبدیل شود.

---

## 1. Data Layer

دو فایل شما عملاً دو نوع اطلاعات می‌دهند.

برای Product Data این فیلدها مستقیم قابل استفاده‌اند:

```text
id
title_fa
Rate
Rate_cnt
Category1
Category2
Brand
Price
Seller
Is_Fake
min_price_last_month
sub_category
```

و برای Review Data مواردی مثل:

```text
body
rate
recommendation_status
is_buyer
advantages
disadvantages
likes
dislikes
product_id
```

برای این قسمت من از:

```python
polars
```

به‌جای Pandas برای پردازش اصلی استفاده می‌کنم.

دلیلش حجم دیتاست است: بیش از یک میلیون محصول و میلیون‌ها نظر دارید.

`Polars + Parquet` برای این پروژه انتخاب خوبی است:

```text
CSV
 ↓
Cleaning
 ↓
Parquet
 ↓
Polars LazyFrame
```

مثلاً:

```python
import polars as pl

products = pl.scan_parquet("products.parquet")
comments = pl.scan_parquet("comments.parquet")
```

برای اجرای سریع‌تر پروژه بهتر است CSV را فقط یک بار بخوانید و بعد همه‌چیز روی Parquet باشد.

در cleaning باید دست‌کم رشته‌های خالی و `nan` را به مقدار گمشده‌ی واقعی تبدیل کرد، ستون‌های `Is_Fake` و `is_buyer` را به Boolean تبدیل کرد، و فیلدهای `advantages` و `disadvantages` را با parser امن از نمایش رشته‌ایِ لیست به داده‌ی ساخت‌یافته تبدیل کرد. برای اجرای کد رشته‌ای نباید از `eval` استفاده شود. تاریخ فارسی نیز فقط در صورتی به تاریخ استاندارد تبدیل شود که برای وزن‌دهی زمانی لازم باشد.

---

## 2. Persian preprocessing

این بخش مهم است و نباید صرفاً `strip()` انجام دهید.

Pipeline پیشنهادی:

```text
raw Persian text
       ↓
Unicode normalization
       ↓
ي → ی
ك → ک
       ↓
whitespace normalization
       ↓
URL / emoji normalization
       ↓
duplicate removal
       ↓
low-information review filtering
```

مثلاً نظرهایی مثل:

```text
عالی
خوبه
بد نبود
👍
```

نباید همان وزن نظر مفصل خریدار واقعی را داشته باشند.

بااین‌حال حذف یا کم‌وزن‌کردن این نظرها نباید درصدهای خام را پنهان کند. سیستم باید هم «آمار خامِ همه‌ی نظرهای واجدشرایط» و هم، در صورت استفاده، «سیگنال وزن‌دار» را جداگانه نگه دارد.

---

## 3. Product Resolver

قبل از مقایسه باید بفهمیم کاربر دقیقاً کدام محصول‌ها را می‌گوید.

مثلاً:

```text
گوشی A55 و Redmi Note 13 Pro رو مقایسه کن
```

باید تبدیل شود به:

```json
{
  "products": [
    {"id": 12345},
    {"id": 98421}
  ]
}
```

برای این بخش ابتدا نیازی به LLM نیست.

ترکیبی از:

```text
RapidFuzz
+
normalized title search
+
brand matching
```

کافی است.

Python:

```bash
pip install rapidfuzz
```

اگر چند نتیجه‌ی مبهم پیدا شد، آنجا LLM یا clarification وارد شود.

---

## 4. Review Retrieval

این مهم‌ترین قسمت معماری است.

اشتباه رایج این است که مثلاً 100 نظر تصادفی از هر محصول برداریم و بدهیم به GPT.

من پیشنهاد می‌کنم:

```text
Hybrid Retrieval
=
Dense Retrieval
+
Sparse Retrieval
+
Metadata Filtering
```

### Dense Retrieval

برای فارسی فعلاً انتخاب عملی و کم‌ریسک من:

```text
BAAI/bge-m3
```

است.

BGE-M3 چندزبانه است و برای Dense/Sparse Retrieval مناسب است.

مثلاً:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

embeddings = model.encode(
    reviews,
    return_dense=True,
    return_sparse=True
)
```

---

## 5. Vector Database

دو انتخاب دارید.

برای پروژه‌ی Bootcamp:

```text
FAISS
```

برای **Dense-only retrieval** کافی است؛ اما به‌تنهایی Sparse یا Hybrid Search و metadata filtering را فراهم نمی‌کند. در این انتخاب باید BM25/یک sparse index و فیلتر متادیتا نیز جداگانه پیاده‌سازی شوند.

اما اگر بخواهید معماری تمیزتر و قابل گسترش‌تری داشته باشید:

```text
Qdrant
```

را ترجیح می‌دهم.

مزیت Qdrant برای این پروژه این است که metadata filtering بسیار راحت می‌شود:

```json
{
  "product_id": 43132,
  "is_buyer": true,
  "rate": 2
}
```

پس query می‌تواند چیزی مثل این باشد:

```text
"مشکل باتری این محصول چیست؟"

filter:
product_id = 43132
```

یعنی همه‌ی corpus را جست‌وجو نمی‌کنیم.

پس انتخاب اجرایی باید صریح باشد: یا «FAISS + BM25 + فیلتر در Python» برای نسخه‌ی سبک‌تر، یا «Qdrant با dense/sparse و metadata filtering» برای Hybrid واقعی. این دو گزینه نباید معادل فرض شوند.

---

## 6. Hybrid Retrieval

Pipeline:

```text
User question
      ↓
Dense search
      ↓
Sparse search
      ↓
Merge
      ↓
Top 50
```

مثلاً برای:

```text
از نظر کیفیت ساخت کدام بهتر است؟
```

برای هر محصول جداگانه:

```text
Product A reviews
       ↓
quality / build retrieval
       ↓
50 reviews

Product B reviews
       ↓
quality / build retrieval
       ↓
50 reviews
```

سپس reranker.

**مرز مهم:** خروجی Top-K این مرحله فقط برای پیدا کردن شواهد متنیِ قابل‌نمایش و مرتبط با سؤال است. محاسبه‌ی درصد پیشنهاد خرید، فراوانی جنبه‌ها و نتیجه‌ی کلی رضایت نباید تنها بر اساس این Top-K انجام شود، چون retrieval سوگیری ایجاد می‌کند. این آمار باید از همه‌ی نظرهای واجدشرایط محصول، یا از نمونه‌ی تصادفی/طبقه‌بندی‌شده‌ی ازپیش‌تعریف‌شده و قابل دفاع، ساخته شود.

---

## 7. Reranker

اینجا یکی از مهم‌ترین بهبودهای پروژه است.

Embedding برای **candidate retrieval** مناسب است، ولی دقیق‌ترین انتخاب شواهد نیست.

پس:

```text
100,000 reviews
      ↓
Embedding retrieval
      ↓
Top 50
      ↓
Cross Encoder Reranker
      ↓
Top 8–15
```

مدلی که پیشنهاد می‌کنم:

```text
BAAI/bge-reranker-v2-m3
```

چون multilingual و نسبتاً سبک است.

---

## 8. Review Weighting

تمام نظرات نباید وزن یکسان داشته باشند.

من برای هر review یک وزن می‌سازم:

```text
review_weight =
buyer_weight
× helpfulness_weight
× text_quality_weight
× recency_weight
```

مثلاً:

```python
weight = (
    1.3 if is_buyer else 1.0
) * (
    1 + log1p(likes)
)
```

بعد:

```text
verified buyer
+
20 likes
+
متن توضیحی مفصل
```

وزن بیشتری از:

```text
"خوبه"
```

خواهد داشت.

وزن‌دهی یک سیگنال مکمل است، نه جایگزین آمار خام. برای نمونه، ابتدا این مقادیر گزارش می‌شوند:

```text
recommended_pct = recommended / valid_recommendation_status
not_recommended_pct = not_recommended / valid_recommendation_status
no_idea_pct = no_idea / valid_recommendation_status
opinionated_recommend_pct = recommended / (recommended + not_recommended)
```

در کنار هر درصد، شمار مطلق نیز می‌آید. سپس، در صورت نیاز، نسخه‌ی وزن‌دار با برچسب روشن «سیگنال وزن‌دار» ارائه می‌شود. به همین ترتیب، برای هر Aspect درصد مثبت/منفی و حجم شواهد محاسبه می‌شود؛ اگر اختلاف درصد ناچیز یا حجم نمونه کم باشد، خروجی باید آن را نامطمئن اعلام کند.

---

## 9. Aspect Extraction

اینجا به جای summary ساده، نظرات را به **Aspect** تبدیل می‌کنیم.

مثلاً موبایل:

```text
battery
camera
display
performance
build_quality
charging
price_value
```

کفش:

```text
comfort
size
material
durability
appearance
```

یعنی Aspectها باید category-aware باشند.

خروجی هر review می‌تواند این باشد:

```json
{
  "review_id": 928372,
  "aspects": [
    {
      "name": "battery",
      "sentiment": "negative",
      "evidence": "باتری نهایت یک روز دوام میاره"
    }
  ]
}
```

برای MVP می‌توانید این استخراج را با LLM انجام دهید.

اما نتیجه را **cache** کنید؛ نباید در هر query همه‌ی reviewها دوباره به LLM فرستاده شوند.

با توجه به بیش از شش میلیون نظر و سقف ۵ دلار API، استخراج Aspect با LLM برای کل دیتاست گزینه‌ی پیش‌فرض نیست. راه عملی‌تر یکی از این‌هاست: مدل محلی، واژگان/قواعد اولیه، پردازش آفلاین فقط برای محصولات منتخب، یا نمونه‌گیری قابل دفاع. هر انتخاب باید پوشش داده و هزینه‌ی آن را گزارش کند.

---

## 10. Precompute کردن Review Intelligence

پیشنهاد می‌کنم یک مرحله Offline داشته باشید:

```text
comments.csv
     ↓
cleaning
     ↓
embedding
     ↓
aspect extraction
     ↓
sentiment
     ↓
review quality
     ↓
stored features
```

و در runtime:

```text
query
 ↓
retrieve
 ↓
aggregate
 ↓
compare
```

این کار latency و cost را شدیداً کم می‌کند.

---

## 11. Product Review Profile

بعد از preprocessing برای هر محصول یک Profile بسازید.

مثلاً:

```json
{
  "product_id": 12039,

  "stats": {
    "review_count": 382,
    "buyer_review_count": 279,
    "recommendation_counts": {
      "recommended": 241,
      "not_recommended": 57,
      "no_idea": 84
    },
    "recommendation_percentages": {
      "recommended": 0.631,
      "not_recommended": 0.149,
      "no_idea": 0.220,
      "opinionated_recommend": 0.809
    }
  },

  "aspects": {
    "battery": {
      "positive_count": 61,
      "negative_count": 15,
      "neutral_count": 8,
      "positive_pct": 0.726,
      "negative_pct": 0.179,
      "support": 84
    },
    "camera": {
      "positive_count": 63,
      "negative_count": 30,
      "neutral_count": 10,
      "positive_pct": 0.612,
      "negative_pct": 0.291,
      "support": 103
    }
  }
}
```

این خیلی بهتر از این است که LLM خودش از روی 300 review حدس بزند «کدام محصول بهتر است».

`recommendation_percentages` باید فقط روی statusهای معتبر محاسبه شود و سهم `no_idea` نیز جداگانه نمایش داده شود. برای جمع‌بندی «نظرِ دارای موضع»، `opinionated_recommend` از مخرج `recommended + not_recommended` استفاده می‌کند. این تعریف‌ها باید در کد و گزارش ثابت بمانند.

---

## 12. Comparison Engine

این قسمت را **deterministic Python** بنویسید.

مثلاً:

```python
class ProductComparator:

    def compare_price(self, a, b):
        ...

    def compare_rating(self, a, b):
        ...

    def compare_aspect(self, a, b, aspect):
        ...

    def compare_recommendation(self, a, b):
        ...
```

خروجی:

```json
{
  "price": {
    "winner": "A",
    "a": 12000000,
    "b": 14300000
  },

  "battery": {
    "winner": "B",
    "a_positive_pct": 0.63,
    "a_support": 73,
    "b_positive_pct": 0.81,
    "b_support": 109,
    "evidence_ids": [811, 927, 1128]
  }
}
```

این یکی از مهم‌ترین تصمیم‌های معماری است:

**LLM عددها را مقایسه نکند.**

Python مقایسه کند؛ LLM توضیح بدهد.

`winner` نیز نباید همیشه اجباری باشد. اگر حجم شواهد به حداقل تعیین‌شده نرسیده، اختلاف از آستانه‌ی عملی کوچک‌تر است، یا درصدهای مثبت و منفی نزدیک‌اند، مقدار آن باید `inconclusive` باشد. منطق تصمیم و آستانه‌ها باید پیش از ارزیابی ثبت شوند.

---

## 13. LLM

برای generation نیاز به مدل خیلی بزرگ ندارید.

ورودی LLM باید چیزی شبیه این باشد:

```text
PRODUCT_A_STRUCTURED_DATA

PRODUCT_B_STRUCTURED_DATA

COMPARISON_RESULTS

REVIEW_EVIDENCE

USER_QUERY
```

نه:

```text
500 نظر خام
```

برای خروجی هم Structured Output بگیرید.

مثلاً با Pydantic:

```python
class ComparisonResponse(BaseModel):
    direct_facts: list[Fact]
    user_evidence: list[Evidence]
    inference: list[Inference]
    recommendation: str
    confidence: float
```

---

## 14. Output پیشنهادی

مثلاً کاربر می‌پرسد:

> A55 و Redmi Note 13 Pro رو مقایسه کن.

خروجی سیستم:

```text
Samsung A55
14,900,000 تومان
امتیاز محصول: 86 / 100
تعداد رأی: 3,420

Redmi Note 13 Pro
13,200,000 تومان
امتیاز محصول: 84 / 100
تعداد رأی: 2,810
```

سپس:

| معیار | A55 | Note 13 Pro | برنده |
|---|---|---|---|
| قیمت | گران‌تر | ارزان‌تر | Redmi |
| کیفیت ساخت | رضایت بالا | رضایت متوسط | A55 |
| باتری | خوب | بسیار خوب | Redmi |
| دوربین | خوب | رضایت بالاتر | Redmi |

و بعد:

```text
طبق داده مستقیم:
Redmi حدود ۱.۷ میلیون ارزان‌تر است.

طبق نظرات کاربران:
کاربران A55 بیشتر از کیفیت ساخت رضایت داشته‌اند.
[review: 18291, 18372, 19123]

کاربران Redmi بیشتر به شارژدهی باتری اشاره مثبت کرده‌اند.
[review: 92123, 93218, 98127]

استنتاج سیستم:
اگر کیفیت ساخت برای شما مهم‌تر است A55 انتخاب مناسب‌تری است.
اگر قیمت و باتری مهم‌تر است Redmi ارزش خرید بیشتری دارد.
```

عددهای مثال بالا صرفاً نمایشی‌اند. در خروجی واقعی، مقدار `Rate` ابتدا باید با بررسی دامنه‌ی دیتاست تفسیر شود؛ نمونه‌های واقعی مقادیری مانند ۹۰ دارند، پس نباید بدون تبدیل مستند آن را به مقیاس ۵تایی نمایش داد. واحد قیمت هم تا زمان اعتبارسنجی منبع داده، باید با احتیاط و ترجیحاً به‌صورت مقدار خام گزارش شود.

---

## 15. Grounding Validator

یک مرحله‌ی بسیار خوب برای بالا بردن کیفیت:

```text
Generated Answer
       ↓
Claim Extraction
       ↓
Check against:
- product DB
- retrieved reviews
       ↓
unsupported claim?
       ↓
remove / rewrite
```

مثلاً LLM بنویسد:

```text
A55 ضدآب‌تر است
```

ولی دیتاست چنین اطلاعاتی نداشته باشد.

Validator باید آن را رد کند.

اولین خط دفاع باید deterministic باشد: خروجی ساخت‌یافته فقط اجازه‌ی ارجاع به فیلدهای product DB و `review_id`های بازیابی‌شده را داشته باشد و Python وجود و سازگاری آن‌ها را بررسی کند. اعتبارسنجی LLM می‌تواند یک لایه‌ی کمکی باشد، اما تضمین grounding محسوب نمی‌شود و هزینه‌اش باید اندازه‌گیری شود.

---

## 16. Stack پیشنهادی Python

من برای اجرای پروژه این stack را انتخاب می‌کنم:

```text
Python 3.12+

Data:
Polars
PyArrow
Parquet

Persian preprocessing:
regex
hazm / dadmatools فقط در صورت نیاز

Search:
Qdrant
یا
FAISS

Embedding:
BAAI/bge-m3

Reranker:
BAAI/bge-reranker-v2-m3

LLM:
API-based LLM
با Structured Output

Schema:
Pydantic v2

API:
FastAPI

Cache:
Redis
یا برای Bootcamp:
diskcache

Experiment:
Jupyter
MLflow اختیاری

Evaluation:
Ragas-style metrics
Custom retrieval metrics
LLM-as-a-Judge
Human evaluation
```

---

## 17. LangChain یا LlamaIndex؟

برای این بخش من **در هسته‌ی پروژه از LangChain استفاده نمی‌کنم**.

چون Pipeline شما مشخص است و dependency/abstraction اضافه احتمالاً فقط debugging را سخت می‌کند.

ترجیح من:

```text
Python
+
Pydantic
+
Qdrant client
+
model SDK
```

است.

اگر بخواهیم workflow حالت Agentic پیدا کند، آن‌وقت:

```text
LangGraph
```

انتخاب خوبی است.

ولی برای:

```text
retrieve → rerank → aggregate → compare → generate
```

Agent اساساً لازم نیست.

---

## 18. پیشنهاد برای ارزیابی و گرفتن امتیاز بهتر

من سه نسخه پیاده می‌کنم.

### Baseline

```text
BM25
   ↓
Top reviews
   ↓
LLM
```

### نسخه دوم

```text
Dense Embedding
   ↓
Top reviews
   ↓
LLM
```

### نسخه نهایی

```text
Hybrid Search
      ↓
Reranker
      ↓
Aspect Aggregation
      ↓
Comparison Engine
      ↓
LLM
```

در هر سه نسخه، آمارهای مقایسه‌ایِ خام (شمارش و درصد recommendation/aspect) باید از یک تعریف داده‌ای مشترک ساخته شوند؛ تفاوت آزمایش‌ها فقط باید در کیفیت بازیابی و انتخاب شواهد باشد. در غیر این صورت مقایسه‌ی کیفیت پاسخ منصفانه نخواهد بود.

بعد مقایسه می‌کنیم:

```text
Recall@K
NDCG@K
Grounding
Answer quality
Latency
Token Cost
```

---

## جمع‌بندی معماری پیشنهادی

```text
                 OFFLINE PIPELINE
                       │
CSV
 │
 ▼
Polars Cleaning
 │
 ▼
Parquet
 │
 ├──────────► Product Structured DB
 │
 ▼
Persian Review Normalization
 │
 ▼
BGE-M3 Embeddings
 │
 ▼
Qdrant
 │
 ▼
Aspect / Sentiment Extraction
 │
 ▼
Product Review Profiles


                 ONLINE PIPELINE

User
 │
 ▼
Query Parser
 │
 ▼
Product Resolver
 │
 ├──────────────► Structured Product Data
 │
 ▼
Hybrid Retrieval
 │
 ▼
BGE-Reranker-v2-M3
 │
 ▼
Evidence Aggregation
 │
 ▼
Product Comparator
│  ├── درصدهای خام + حجم نمونه
│  ├── سیگنال وزن‌دار (اختیاری و برچسب‌دار)
│  └── inconclusive در شواهد ناکافی/متناقض
 │
 ▼
LLM Structured Generation
 │
 ▼
Grounding Validator
 │
 ▼
Answer + Evidence IDs
```

از نظر من **این بهترین نقطه‌ی شروع برای پروژه‌ی شماست**؛ نه بیش‌ازحد ساده است و نه به سمت معماری Agentic غیرضروری می‌رود.

---

## مواردی که برای طراحی دقیق‌تر باید مشخص شوند

دو مورد هنوز برای طراحی نهایی مهم هستند:

1. آیا مسئولیت شما فقط **بخش ۳ مقایسه محصولات** است یا این ماژول باید روی خروجی بخش‌های ۱ و ۲ گروه سوار شود؟
2. آیا GPU در اختیار دارید یا باید کل پروژه روی CPU/Colab اجرا شود؟

پاسخ این دو مورد تعیین می‌کند Qdrant/FAISS، embedding، batch size و شکل precomputation را چگونه بچینیم.
