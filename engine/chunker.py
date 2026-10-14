from typing import List, Dict, Any
from engine.models import ProductItem, PolicyItem

class DocumentChunk:
    def __init__(self, doc_id: str, title: str, doc_type: str, text: str, metadata: Dict[str, Any], citation_tag: str):
        self.doc_id = doc_id
        self.title = title
        self.doc_type = doc_type
        self.text = text
        self.metadata = metadata
        self.citation_tag = citation_tag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "text": self.text,
            "metadata": self.metadata,
            "citation_tag": self.citation_tag
        }

class ECommerceChunker:
    """
    Context-Enriched Domain Chunker for E-Commerce RAG.
    Prepend rich metadata headers to each semantic chunk to guarantee high-precision
    lexical (BM25) and semantic (Dense) retrieval without context loss.
    """
    def chunk_product(self, product: ProductItem) -> List[DocumentChunk]:
        chunks = []
        
        # Header prepended to prevent loss of metadata in embeddings
        metadata_header = (
            f"[PRODUCT: {product.title} | Brand: {product.brand} | Category: {product.category} | "
            f"Price: ${product.price:.2f} | Stock: {'In-Stock' if product.in_stock else 'Out-of-Stock'} | "
            f"Skin: {', '.join(product.skin_types)}]"
        )
        
        # Primary Overview Chunk (Benefits + Clinical Claims)
        overview_text = (
            f"{metadata_header}\n"
            f"Summary: Clinical {product.category} formulated for {', '.join(product.skin_types)} skin.\n"
            f"Primary Benefits:\n- " + "\n- ".join(product.benefits) + "\n"
            f"Key Actives: {', '.join(product.key_ingredients)}.\n"
            f"Rating: {product.rating}/5.0 ({product.review_count} reviews)."
        )
        chunks.append(DocumentChunk(
            doc_id=product.id,
            title=product.title,
            doc_type="product_overview",
            text=overview_text,
            metadata={
                "product_id": product.id,
                "price": product.price,
                "category": product.category,
                "skin_types": product.skin_types,
                "in_stock": product.in_stock,
                "ingredients": product.key_ingredients
            },
            citation_tag=product.doc_citation
        ))

        # Secondary Technical Chunk (Usage, Contraindications, Protocol)
        contraindications_text = "; ".join(product.contraindications) if product.contraindications else "None noted. Safe for sensitive skin."
        usage_text = (
            f"{metadata_header}\n"
            f"Application Protocol: {product.usage}\n"
            f"Safety & Contraindications: {contraindications_text}\n"
            f"Clinical Document Reference: {product.doc_citation}\n"
            f"Suggested Routine Complements: {', '.join(product.bundle_affinities)}"
        )
        chunks.append(DocumentChunk(
            doc_id=f"{product.id}_USAGE",
            title=f"{product.title} (Protocol & Safety)",
            doc_type="product_safety",
            text=usage_text,
            metadata={
                "product_id": product.id,
                "price": product.price,
                "category": product.category,
                "skin_types": product.skin_types,
                "in_stock": product.in_stock,
                "contraindications": product.contraindications
            },
            citation_tag=product.doc_citation
        ))
        
        return chunks

    def chunk_policy(self, policy: PolicyItem) -> DocumentChunk:
        policy_header = f"[STORE POLICY: {policy.topic} | Citation Tag: {policy.citation_tag}]"
        text = f"{policy_header}\nSummary: {policy.summary}\nFull Terms: {policy.content}"
        return DocumentChunk(
            doc_id=policy.id,
            title=policy.topic,
            doc_type="policy",
            text=text,
            metadata={
                "policy_id": policy.id,
                "topic": policy.topic,
                "citation_tag": policy.citation_tag
            },
            citation_tag=policy.citation_tag
        )
