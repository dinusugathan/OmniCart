import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from engine.models import (
    ProductItem, PolicyItem, ExtractedFilters, ProductRecommendation,
    BundleOffer, GuardrailAlert, RecommendationResponse
)
from engine.chunker import ECommerceChunker, DocumentChunk
from engine.hybrid_retriever import HybridRetriever
from engine.reranker import ProductReranker
from engine.tools import ECommerceTools
from engine.guardrails import SafetyGuardrails

class OmniCartAgent:
    """
    OmniCart E-Commerce Recommendation & Grounded RAG Agent.
    Pipeline:
      1. Query Filter Extraction & Intent Parsing
      2. Input Guardrail Inspection
      3. Hybrid Retrieval (BM25 + Dense + RRF + Filters)
      4. Cross-Encoder Reranking (FlashRank)
      5. Safety & Contraindication Guardrails
      6. Deterministic Tool Execution (Dynamic Bundling & AOV lift)
      7. Grounded Output Synthesis with Verified Citations
    """
    def __init__(self, catalog_path: str, policies_path: str):
        self.catalog_path = catalog_path
        self.policies_path = policies_path
        self.chunker = ECommerceChunker()
        self.guardrails = SafetyGuardrails()
        self.reranker = ProductReranker()
        
        self.products: Dict[str, ProductItem] = {}
        self.policies: Dict[str, PolicyItem] = {}
        self.chunks: List[DocumentChunk] = []
        
        self._load_data()
        self.tools = ECommerceTools(self.products)
        self.retriever = HybridRetriever(self.chunks)

    def _load_data(self):
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            raw_products = json.load(f)
            for item in raw_products:
                prod = ProductItem(**item)
                self.products[prod.id] = prod
                self.chunks.extend(self.chunker.chunk_product(prod))

        with open(self.policies_path, "r", encoding="utf-8") as f:
            raw_policies = json.load(f)
            for item in raw_policies:
                pol = PolicyItem(**item)
                self.policies[pol.id] = pol
                self.chunks.append(self.chunker.chunk_policy(pol))

    def extract_filters(self, query: str) -> ExtractedFilters:
        """
        Deterministic / regex-based extractor for budget, skin types, categories, and policy intent.
        """
        q_lower = query.lower()
        filters = ExtractedFilters()

        # 1. Budget extraction (e.g. "under $50", "below 40", "under 60 dollars", "<$30")
        price_match = re.search(r"(?:under|below|less than|max|budget of)\s*(?:\$)?\s*(\d+(?:\.\d+)?)", q_lower)
        if price_match:
            filters.max_price = float(price_match.group(1))

        # 2. Skin Type extraction
        skin_keywords = {
            "acne": "Acne-prone",
            "blemish": "Acne-prone",
            "sensitive": "Sensitive",
            "dry": "Dry",
            "oily": "Oily",
            "combination": "Combination",
            "dull": "Dull",
            "aging": "Aging",
            "eczema": "Eczema-prone"
        }
        for kw, val in skin_keywords.items():
            if kw in q_lower:
                filters.skin_type = val
                break

        # 3. Category extraction (ignore single category constraint if asking for routine/regimen/steps)
        is_routine_request = any(r in q_lower for r in ["routine", "regimen", "system", "bundle", "steps", "and"])
        if not is_routine_request:
            cat_keywords = {
                "cleanser": "Cleanser",
                "wash": "Cleanser",
                "balm": "Cleanser",
                "serum": "Serum",
                "elixir": "Serum",
                "moisturizer": "Moisturizer",
                "cream": "Moisturizer",
                "sunscreen": "Sunscreen",
                "spf": "Sunscreen",
                "oil": "Facial Oil",
                "exfoliant": "Exfoliant",
                "peel": "Exfoliant",
                "eye": "Eye Care"
            }
            for kw, val in cat_keywords.items():
                if kw in q_lower:
                    filters.category = val
                    break

        # 4. Policy query detection
        policy_keywords = ["return", "refund", "guarantee", "shipping", "deliver", "policy", "bundle discount", "discount"]
        if any(pk in q_lower for pk in policy_keywords):
            filters.is_policy_query = True

        return filters

    def process_query(self, query: str) -> RecommendationResponse:
        start_time = time.time()

        # Step 1: Input Guardrail Check
        input_guardrail = self.guardrails.check_input_safety(query)
        if input_guardrail.triggered:
            return RecommendationResponse(
                query=query,
                grounded_summary=input_guardrail.message or "Request blocked by safety policy.",
                guardrail=input_guardrail,
                retrieval_stats={"latency_ms": round((time.time() - start_time) * 1000, 2)}
            )

        # Step 2: Filter Extraction
        filters = self.extract_filters(query)

        # Step 3: Hybrid Retrieval (Sparse + Dense + RRF + Metadata Filters)
        retrieved_candidates = self.retriever.retrieve(query, filters=filters, top_k=8)

        # Step 4: Cross-Encoder Reranking
        reranked_chunks = self.reranker.rerank(query, retrieved_candidates, top_n=5)

        # Step 5: Clinical Contraindication Guardrail
        contra_guardrail = self.guardrails.check_contraindications(query, reranked_chunks)

        # Step 6: Identify Recommended Products & Citations
        recommended_products: List[ProductRecommendation] = []
        policy_citations: List[str] = []
        seen_pids = set()

        for chunk, score in reranked_chunks:
            if chunk.doc_type in ["product_overview", "product_safety"]:
                pid = chunk.metadata.get("product_id")
                if pid and pid in self.products and pid not in seen_pids:
                    prod = self.products[pid]
                    seen_pids.add(pid)
                    recommended_products.append(ProductRecommendation(
                        id=prod.id,
                        title=prod.title,
                        brand=prod.brand,
                        category=prod.category,
                        price=prod.price,
                        reason=f"Formulated for {', '.join(prod.skin_types)} skin with {', '.join(prod.key_ingredients[:2])}. Highly rated ({prod.rating}★).",
                        key_ingredients=prod.key_ingredients,
                        doc_citation=prod.doc_citation
                    ))
            elif chunk.doc_type == "policy":
                cit = chunk.metadata.get("citation_tag")
                if cit and cit not in policy_citations:
                    policy_citations.append(cit)

        # Step 7: Deterministic Tool Calling for Dynamic Bundling (AOV Lift)
        bundle_offer: Optional[BundleOffer] = None
        if len(recommended_products) >= 2:
            bundle_pids = [p.id for p in recommended_products[:3]]
            bundle_offer = self.tools.calculate_bundle_pricing(bundle_pids)

        # Step 8: Build Grounded Summary
        summary_lines = []
        if contra_guardrail.triggered:
            summary_lines.append(f"⚠️ **Safety Notice**: {contra_guardrail.message}\n")

        if recommended_products:
            total_price = sum(p.price for p in recommended_products)
            summary_lines.append(f"Here is your personalized routine based on your criteria:")
            for p in recommended_products:
                summary_lines.append(f"• **{p.title}** (${p.price:.2f}) — {p.reason} [{p.doc_citation}]")
            
            if bundle_offer:
                summary_lines.append(
                    f"\n✨ **Smart Routine Offer**: Bundle all {len(bundle_offer.included_product_ids)} items "
                    f"for **${bundle_offer.discounted_total:.2f}** (normally ${bundle_offer.original_total:.2f}, saving ${bundle_offer.savings_amount:.2f} with our {bundle_offer.discount_percent:.0f}% regimen discount)."
                )
        elif policy_citations:
            # Policy-specific response
            policy_texts = [f"• **{self.policies[pol_id].topic}**: {self.policies[pol_id].content} [{self.policies[pol_id].citation_tag}]" for pol_id in self.policies if self.policies[pol_id].citation_tag in policy_citations]
            summary_lines.append("Here is the relevant store policy information:")
            summary_lines.extend(policy_texts)
        else:
            summary_lines.append("No exact in-stock match found within your specified criteria. Try broadening your price or skin type filter.")

        # Step 9: Verify Groundedness
        groundedness = self.guardrails.verify_groundedness(recommended_products, reranked_chunks)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        applied_filters_dict = {
            "max_price": filters.max_price,
            "skin_type": filters.skin_type,
            "category": filters.category,
            "in_stock_only": filters.in_stock_only
        }

        return RecommendationResponse(
            query=query,
            grounded_summary="\n".join(summary_lines),
            recommended_products=recommended_products,
            bundle_offer=bundle_offer,
            policy_citations=policy_citations,
            guardrail=contra_guardrail if contra_guardrail.triggered else input_guardrail,
            metadata_filters_applied=applied_filters_dict,
            retrieval_stats={
                "retrieved_candidates_count": len(retrieved_candidates),
                "reranked_count": len(reranked_chunks),
                "groundedness_score": groundedness,
                "latency_ms": latency_ms
            }
        )
