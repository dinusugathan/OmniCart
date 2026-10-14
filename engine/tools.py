from typing import List, Dict, Any, Optional
from engine.models import ProductItem, BundleOffer

class ECommerceTools:
    """
    Deterministic Tool Suite for E-Commerce Recommendation & AOV Optimization.
    Eliminates pricing and discount hallucinations by computing deterministic numbers.
    """
    def __init__(self, catalog: Dict[str, ProductItem]):
        self.catalog = catalog

    def calculate_bundle_pricing(self, product_ids: List[str]) -> Optional[BundleOffer]:
        """
        Deterministic tool to calculate tiered bundle pricing:
        - 2 complementary steps: 10% discount
        - 3 complementary steps: 15% discount
        - 4+ complementary steps: 20% discount
        """
        valid_products: List[ProductItem] = []
        for pid in product_ids:
            if pid in self.catalog and self.catalog[pid].in_stock:
                valid_products.append(self.catalog[pid])

        if len(valid_products) < 2:
            return None

        # Tiered discount policy
        count = len(valid_products)
        if count == 2:
            discount_pct = 10.0
            bundle_name = "Essential 2-Step Duo"
        elif count == 3:
            discount_pct = 15.0
            bundle_name = "Core 3-Step Treatment Regimen"
        else:
            discount_pct = 20.0
            bundle_name = "Complete Clinical 4-Step System"

        original_sum = sum(p.price for p in valid_products)
        discount_amount = round(original_sum * (discount_pct / 100.0), 2)
        discounted_price = round(original_sum - discount_amount, 2)
        
        # AOV (Average Order Value) lift rationale
        cheapest_item = min(valid_products, key=lambda p: p.price)
        aov_pitch = (
            f"Bundle & Save: Get {discount_pct:.0f}% off when purchasing this curated regimen together. "
            f"You save ${discount_amount:.2f} instantly (equivalent to nearly {discount_pct:.0f}% of your {cheapest_item.category})."
        )

        return BundleOffer(
            bundle_title=bundle_name,
            included_product_ids=[p.id for p in valid_products],
            product_names=[p.title for p in valid_products],
            original_total=round(original_sum, 2),
            discount_percent=discount_pct,
            discounted_total=discounted_price,
            savings_amount=discount_amount,
            aov_pitch=aov_pitch
        )

    def check_ingredient_compatibility(self, ingredients_a: List[str], ingredients_b: List[str]) -> Optional[str]:
        """
        Deterministic contraindication rule checker between two sets of active ingredients.
        """
        a_str = " ".join(ingredients_a).lower()
        b_str = " ".join(ingredients_b).lower()

        # Rule 1: Retinoids + Direct AHA/BHA
        has_retinoid = any(r in a_str or r in b_str for r in ["retinal", "retinol", "retinoid"])
        has_acid = any(ac in a_str or ac in b_str for ac in ["glycolic", "lactic", "salicylic", "aha 7%"])
        
        if has_retinoid and has_acid:
            return (
                "CLINICAL CONTRAINDICATION: Combining Retinal/Retinol with strong AHA/BHA exfoliants "
                "in the same nighttime routine can strip the lipid barrier and cause severe contact dermatitis. "
                "Alternate nights or use AHA in the evening and Retinal 48 hours later."
            )

        # Rule 2: Vitamin C (Pure L-Ascorbic Acid) + Copper Peptides
        has_copper = "copper" in a_str or "copper" in b_str
        has_vit_c = "l-ascorbic" in a_str or "vitamin c" in a_str or "l-ascorbic" in b_str or "vitamin c" in b_str
        
        if has_copper and has_vit_c:
            return (
                "CLINICAL CONTRAINDICATION: Pure L-Ascorbic Acid oxidizes Copper Peptides, neutralizing their efficacy. "
                "Use Vitamin C in your morning routine and Copper Peptides at night."
            )

        return None
