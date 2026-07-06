from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, EvaluationConfig, RetrievalConfig, LLMConfig
from llm import create_llm, llm_invoke
from evaluation_judge import create_judge
from retriever import retrieve_documents
from llm_chain import generate_answer


def run_retrieval(
    question: str,
    retrieval_cfg: RetrievalConfig,
) -> Dict[str, Any]:
    start = time.time()
    documents, scores = retrieve_documents(question, retrieval_cfg)
    elapsed = time.time() - start

    return {
        "question": question,
        "num_chunks": len(documents),
        "retrieval_time_s": round(elapsed, 3),
        "scores": scores,
        "documents": documents,
    }


def run_generation(
    question: str,
    documents,
    llm_cfg: LLMConfig,
) -> Dict[str, Any]:
    start = time.time()
    answer = generate_answer(question, documents, llm_cfg)
    elapsed = time.time() - start

    return {
        "answer": answer,
        "generation_time_s": round(elapsed, 3),
    }


def run_full_pipeline(
    question: str,
    retrieval_cfg: RetrievalConfig,
    llm_cfg: LLMConfig,
) -> Dict[str, Any]:
    retrieval = run_retrieval(question, retrieval_cfg)
    generation = run_generation(question, retrieval["documents"], llm_cfg)

    return {
        "question": question,
        "answer": generation["answer"],
        "num_chunks": retrieval["num_chunks"],
        "retrieval_time_s": retrieval["retrieval_time_s"],
        "generation_time_s": generation["generation_time_s"],
        "total_time_s": round(
            retrieval["retrieval_time_s"] + generation["generation_time_s"], 3
        ),
        "scores": retrieval["scores"],
        "documents": retrieval["documents"],
    }


def run_deepeval_metrics(
    test_case,
    judge,
    metric_names: Optional[List[str]] = None,
    threshold: float = 0.75,
) -> Dict[str, float]:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )

    available = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
    }

    if metric_names is None:
        metric_names = ["faithfulness", "answer_relevancy"]

    scores = {}
    for name in metric_names:
        if name not in available:
            continue
        metric = available[name](
            threshold=threshold,
            model=judge,
            include_reason=True,
        )
        if name in ("contextual_precision", "contextual_recall"):
            metric.required_multiple = False
        try:
            metric.measure(test_case)
            scores[name] = round(metric.score, 4)
            scores[f"{name}_reason"] = metric.reason or ""
            scores[f"{name}_passed"] = 1 if metric.score >= threshold else 0
        except Exception as exc:
            scores[name] = 0.0
            scores[f"{name}_reason"] = str(exc)
            scores[f"{name}_passed"] = 0

    return scores
