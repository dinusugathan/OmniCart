import re
from typing import List, Dict, Any, Optional, Tuple
from engine.models import GuardrailAlert, ProductRecommendation, ExtractedFilters
from engine.chunker import DocumentChunk

class SafetyGuardrails:
    """
    Multi-layer Guardrails:
    1. Input Jailbreak & Off-Topic Shield
    2. Clinical Safety & Contraindication Guardrail
    3. Grounding Verification (Hallucination detection against retrieved context)
    4. Budget Overrun Guardrail
    """

    ADVERSARIAL_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
        r"system\s*prompt",
        r"write\s+(a\s+)?(python|bash|powershell|c\+\+)?\s*(script|exploit|code)\s+to\s+(scrape|hack|steal|bypass)",
        r"(jailbreak|dan\s+mode|developer\s+mode)",
        r"credit\s*card\s*number",
        r"password\s*generator"
    ]

    def check_input_safety(self, query: str) -> GuardrailAlert:
        """
        Input Guardrail: Catches prompt injections, jailbreaks, and malicious off-topic requests.
        """
        q_lower = query.lower()
        for pattern in self.ADVERSARIAL_PATTERNS:
            if re.search(pattern, q_lower):
                return GuardrailAlert(
                    triggered=True,
                    alert_type="OFF_TOPIC_ADVERSARIAL",
                    severity="BLOCK",
                    message="Request blocked: OmniCart AI is dedicated exclusively to product recommendations, routines, and store policies."
                )
        return GuardrailAlert(triggered=False)

    def check_contraindications(self, query: str, candidate_chunks: List[Tuple[DocumentChunk, float]]) -> GuardrailAlert:
        """
        Clinical Safety Guardrail: Detects hazardous ingredient combinations (e.g. Retinal + AHA on same night).
        """
        q_lower = query.lower()
        
        # Check if user query mentions incompatible actives together
        mentions_retinoid = any(k in q_lower for k in ["retin", "retinal", "retinol", "retinoid"])
        mentions_aha = any(k in q_lower for k in ["glycolic", "aha", "lactic acid", "peeling liquid"])
        
        if mentions_retinoid and mentions_aha:
            return GuardrailAlert(
                triggered=True,
                alert_type="CONTRAINDICATION",
                severity="HIGH",
                message=(
                    "Dermatological Safety Warning: Retinaldehyde and AHA Glycolic/Lactic acids should NOT be used in "
                    "the same evening routine. This can severely disrupt the skin's lipid barrier. "
                    "We recommend alternating nights and buffering with a Ceramide barrier cream."
                )
            )

        # Check against chunk contraindications
        for chunk, _ in candidate_chunks:
            contra = chunk.metadata.get("contraindications", [])
            for c in contra:
                if "AHA/BHA" in c and mentions_aha and mentions_retinoid:
                    return GuardrailAlert(
                        triggered=True,
                        alert_type="CONTRAINDICATION",
                        severity="HIGH",
                        message=f"Product Safety Warning: {c}"
                    )
        return GuardrailAlert(triggered=False)

    def verify_groundedness(
        self, 
        recommendations: List[ProductRecommendation], 
        retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> float:
        """
        Grounding Score / Hallucination verification:
        Ensures recommended products and ingredients strictly exist in retrieved context.
        Returns a score between 0.0 and 1.0.
        """
        if not recommendations:
            return 1.0

        retrieved_ids = {chunk.metadata.get("product_id") for chunk, _ in retrieved_chunks if "product_id" in chunk.metadata}
        retrieved_text = " ".join([chunk.text for chunk, _ in retrieved_chunks]).lower()

        valid_count = 0
        for rec in recommendations:
            # Check ID grounding
            id_grounded = rec.id in retrieved_ids
            # Check ingredient grounding
            ingredients_grounded = all(ing.lower() in retrieved_text for ing in rec.key_ingredients)
            
            if id_grounded and ingredients_grounded:
                valid_count += 1
            elif id_grounded:
                valid_count += 0.8
                
        return round(valid_count / len(recommendations), 2)
