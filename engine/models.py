from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ProductItem(BaseModel):
    id: str
    title: str
    brand: str
    category: str
    price: float
    currency: str = "USD"
    in_stock: bool
    rating: float
    review_count: int
    skin_types: List[str]
    key_ingredients: List[str]
    benefits: List[str]
    contraindications: List[str] = Field(default_factory=list)
    usage: str
    doc_citation: str
    bundle_affinities: List[str] = Field(default_factory=list)

class PolicyItem(BaseModel):
    id: str
    topic: str
    summary: str
    content: str
    citation_tag: str

class ExtractedFilters(BaseModel):
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    skin_type: Optional[str] = None
    in_stock_only: bool = True
    active_ingredients: List[str] = Field(default_factory=list)
    is_policy_query: bool = False
    is_jailbreak_attempt: bool = False

class ProductRecommendation(BaseModel):
    id: str
    title: str
    brand: str
    category: str
    price: float
    reason: str
    key_ingredients: List[str]
    doc_citation: str

class BundleOffer(BaseModel):
    bundle_title: str
    included_product_ids: List[str]
    product_names: List[str]
    original_total: float
    discount_percent: float
    discounted_total: float
    savings_amount: float
    aov_pitch: str

class GuardrailAlert(BaseModel):
    triggered: bool = False
    alert_type: Optional[str] = None  # e.g., "CONTRAINDICATION", "BUDGET_EXCEEDED", "OFF_TOPIC"
    severity: str = "LOW"             # LOW, MEDIUM, HIGH, BLOCK
    message: Optional[str] = None

class RecommendationResponse(BaseModel):
    query: str
    grounded_summary: str
    recommended_products: List[ProductRecommendation] = Field(default_factory=list)
    bundle_offer: Optional[BundleOffer] = None
    policy_citations: List[str] = Field(default_factory=list)
    guardrail: GuardrailAlert = Field(default_factory=GuardrailAlert)
    metadata_filters_applied: Dict[str, Any] = Field(default_factory=dict)
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict)

class EvalCaseResult(BaseModel):
    test_id: str
    query: str
    passed: bool
    retrieval_hit: bool
    mrr_score: float
    groundedness_score: float
    guardrail_correct: bool
    bundle_calculated: bool
    details: str
