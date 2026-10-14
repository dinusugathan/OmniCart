import json
import os
import sys
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.agent import OmniCartAgent
from engine.models import EvalCaseResult

def run_evaluation_benchmark(
    catalog_path: str = "data/products_catalog.json",
    policies_path: str = "data/store_policies.json",
    dataset_path: str = "data/golden_eval_dataset.json",
    output_report_path: str = "evals/benchmark_report.json"
) -> Dict[str, Any]:
    print("=" * 65)
    print("OmniCart AI — Automated Evaluation & Quality Scoring Benchmark")
    print("=" * 65)

    agent = OmniCartAgent(catalog_path, policies_path)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results: List[EvalCaseResult] = []
    total_cases = len(test_cases)
    passed_cases = 0
    mrr_sum = 0.0
    groundedness_sum = 0.0

    for tc in test_cases:
        t_id = tc["id"]
        query = tc["query"]
        expected_skus = tc.get("expected_skus", [])
        expect_contra = tc.get("expected_contraindication_warning", False)
        expect_jailbreak = tc.get("expected_jailbreak_blocked", False)
        expect_citations = tc.get("expected_citations", [])
        expected_filters = tc.get("expected_filters", {})

        print(f"\n[Running {t_id}] Query: \"{query}\"")
        resp = agent.process_query(query)

        # 1. Retrieval & SKU checks
        retrieved_ids = [p.id for p in resp.recommended_products]
        hit = False
        mrr = 0.0
        
        if expected_skus:
            # Check if any expected SKU is retrieved
            for rank, rid in enumerate(retrieved_ids):
                if rid in expected_skus:
                    hit = True
                    mrr = 1.0 / (rank + 1)
                    break
        elif expect_jailbreak or tc.get("expected_citations"):
            # Non-product queries (policy or jailbreak)
            hit = True
            mrr = 1.0

        # 2. Guardrail Check
        guardrail_correct = True
        if expect_jailbreak:
            guardrail_correct = resp.guardrail.triggered and resp.guardrail.alert_type == "OFF_TOPIC_ADVERSARIAL"
        elif expect_contra:
            guardrail_correct = resp.guardrail.triggered and resp.guardrail.alert_type == "CONTRAINDICATION"
        else:
            guardrail_correct = not resp.guardrail.triggered or resp.guardrail.severity != "BLOCK"

        # 3. Groundedness check
        groundedness = resp.retrieval_stats.get("groundedness_score", 1.0)
        
        # 4. Filter adherence check (e.g. price ceiling)
        filter_passed = True
        max_p = expected_filters.get("max_price")
        if max_p is not None:
            for p in resp.recommended_products:
                if p.price > max_p:
                    filter_passed = False

        # 5. Citation check for policy queries
        citation_passed = True
        if expect_citations:
            citation_passed = all(ec in resp.policy_citations for ec in expect_citations)

        # Overall pass criteria for test case
        case_passed = hit and guardrail_correct and filter_passed and citation_passed and (groundedness >= 0.8)

        if case_passed:
            passed_cases += 1
            status_symbol = "[PASS]"
        else:
            status_symbol = "[FAIL]"

        mrr_sum += mrr
        groundedness_sum += groundedness

        details = (
            f"Retrieved: {retrieved_ids} | Guardrail Triggered: {resp.guardrail.triggered} "
            f"({resp.guardrail.alert_type}) | Latency: {resp.retrieval_stats.get('latency_ms', 0)}ms"
        )
        print(f"  Result: {status_symbol} (MRR: {mrr:.2f}, Groundedness: {groundedness:.2f})")
        print(f"  {details}")

        results.append(EvalCaseResult(
            test_id=t_id,
            query=query,
            passed=case_passed,
            retrieval_hit=hit,
            mrr_score=mrr,
            groundedness_score=groundedness,
            guardrail_correct=guardrail_correct,
            bundle_calculated=resp.bundle_offer is not None,
            details=details
        ))

    pass_rate = round((passed_cases / total_cases) * 100, 1)
    mean_mrr = round(mrr_sum / total_cases, 3)
    mean_groundedness = round(groundedness_sum / total_cases, 3)

    summary_report = {
        "total_test_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percentage": pass_rate,
        "mean_reciprocal_rank_mrr": mean_mrr,
        "mean_groundedness_score": mean_groundedness,
        "hallucination_rate": round(1.0 - mean_groundedness, 3),
        "results": [r.model_dump() for r in results]
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 65)
    print("BENCHMARK SCORECARD SUMMARY:")
    print(f"  Total Test Cases : {total_cases}")
    print(f"  Passed Cases     : {passed_cases}/{total_cases} ({pass_rate}%)")
    print(f"  Mean MRR         : {mean_mrr}")
    print(f"  Groundedness     : {mean_groundedness} (1.0 = zero hallucinations)")
    print(f"  Benchmark Report : Saved to {output_report_path}")
    print("=" * 65)

    return summary_report

if __name__ == "__main__":
    run_evaluation_benchmark()
