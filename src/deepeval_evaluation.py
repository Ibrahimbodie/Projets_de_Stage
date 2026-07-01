from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "deepeval"


# --------------------------------------------------
# JUGE OLLAMA POUR DEEPEVAL
# --------------------------------------------------

class OllamaEvaluationJudge(DeepEvalBaseLLM):

    def __init__(
        self,
        model_name: str = "mistral:7b",
    ):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )

            return response.json().get(
                "response", ""
            )

        except Exception as exc:

            return (
                f"Error connecting to Ollama: {exc}"
            )

    async def a_generate(
        self,
        prompt: str,
    ) -> str:

        return self.generate(prompt)

    def get_model_name(self) -> str:

        return self.model_name


# --------------------------------------------------
# CHARGEMENT DU DATASET
# --------------------------------------------------

def load_dataset(
    input_path: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:

    path = Path(input_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":

        df = pd.read_csv(path)

    elif suffix in (".xls", ".xlsx"):

        df = pd.read_excel(path)

    else:

        raise ValueError(
            "Format non supporté. "
            "Utilisez CSV ou Excel."
        )

    if "Question" not in df.columns:

        raise ValueError(
            "Le fichier doit contenir "
            "une colonne 'Question'."
        )

    df = df.dropna(subset=["Question"])

    if limit and limit < len(df):

        df = df.head(limit)

    return df





# --------------------------------------------------
# LISTE DES METRIQUES
# --------------------------------------------------

def build_metrics(
    judge: OllamaEvaluationJudge,
    metric_names: List[str],
    threshold: float = 0.75,
):

    available = {
        "answer_relevancy": (
            AnswerRelevancyMetric
        ),
        "faithfulness": (
            FaithfulnessMetric
        ),
        "contextual_relevancy": (
            ContextualRelevancyMetric
        ),
        "contextual_precision": (
            ContextualPrecisionMetric
        ),
        "contextual_recall": (
            ContextualRecallMetric
        ),
    }

    if "all" in metric_names:

        metric_names = list(
            available.keys()
        )

    metrics = []

    for name in metric_names:

        if name not in available:

            print(
                f"⚠️ Métrique inconnue : "
                f"{name}"
            )

            continue

        metric_class = available[name]

        kwargs = {
            "threshold": threshold,
            "model": judge,
            "include_reason": True,
        }

        if name in (
            "contextual_precision",
            "contextual_recall",
        ):

            kwargs["required_multiple"] = (
                False
            )

        metrics.append(
            metric_class(**kwargs)
        )

    return metrics


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def run_evaluation(
    input_path: str,
    llm_model: str = "mistral:7b",
    judge_model: str = "mistral:7b",
    metrics_list: Optional[List[str]] = None,
    limit: Optional[int] = None,
    threshold: float = 0.75,
):

    if metrics_list is None:

        metrics_list = [
            "faithfulness",
            "answer_relevancy",
        ]

    print(
        f"[🚀] Chargement du dataset : "
        f"{input_path}"
    )

    df = load_dataset(
        input_path,
        limit,
    )

    print(
        f"[📊] {len(df)} questions chargées."
    )

    print(
        f"[🤖] Initialisation du juge : "
        f"{judge_model}"
    )

    judge = OllamaEvaluationJudge(
        model_name=judge_model,
    )

    print(
        f"[🔧] Connexion au pipeline RAG "
        f"(LLM : {llm_model})"
    )

    rag_fn = build_rag_pipeline(
        llm_model=llm_model,
    )

    metrics = build_metrics(
        judge,
        metrics_list,
        threshold,
    )

    print(
        f"[📏] Métriques : "
        f"{[m.__class__.__name__ for m in metrics]}"
    )

    print(
        f"\n{'='*60}\n"
        f"Début de l'évaluation\n"
        f"{'='*60}\n"
    )

    results = []

    for idx, row in df.iterrows():

        question = row["Question"]

        expected_output = (
            row.get(
                "Expected_Output",
                "",
            )
            or row.get(
                "Réponse_Attendue",
                "",
            )
            or ""
        )

        q_id = row.get(
            "ID",
            f"Q{idx+1:02d}",
        )

        niveau = row.get(
            "Niveau_Complexite",
            "",
        )

        print(
            f"[{q_id}] Évaluation... "
            f"{question[:60]}"
        )

        start_time = time.time()

        try:

            actual_output, (
                retrieval_context_list
            ) = rag_fn(question)

        except Exception as exc:

            actual_output = (
                f"Erreur RAG : {exc}"
            )

            retrieval_context_list = [
                "Erreur de récupération."
            ]

        elapsed = round(
            time.time() - start_time,
            2,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=(
                retrieval_context_list
            ),
        )

        row_result = {
            "ID": q_id,
            "Niveau": niveau,
            "Question": question,
            "Expected_Output": (
                expected_output
            ),
            "Actual_Output": actual_output,
            "Response_Time_Sec": elapsed,
        }

        all_passed = True

        for metric in metrics:

            try:

                metric.measure(test_case)

                score = round(
                    metric.score,
                    4,
                )

                reason = (
                    metric.reason
                    or ""
                )

                passed = (
                    1
                    if score >= threshold
                    else 0
                )

                if not passed:

                    all_passed = False

            except Exception as exc:

                score = 0.0
                reason = (
                    f"Erreur métrique : "
                    f"{exc}"
                )
                passed = 0
                all_passed = False

            metric_name = (
                metric.__class__.__name__
            )

            row_result[
                f"{metric_name}_Score"
            ] = score

            row_result[
                f"{metric_name}_Reason"
            ] = reason

            row_result[
                f"{metric_name}_Passed"
            ] = passed

        row_result["All_Passed"] = (
            1 if all_passed else 0
        )

        results.append(row_result)

    return results


# --------------------------------------------------
# EXPORT
# --------------------------------------------------

def export_results(
    results: List[dict],
    suffix: str = "",
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"deepeval_results"
        f"{suffix}"
        f"_{timestamp}.xlsx"
    )

    output_path = (
        RESULTS_DIR / filename
    )

    df = pd.DataFrame(results)

    df.to_excel(
        output_path,
        index=False,
    )

    print(
        f"\n[✅] Résultats exportés : "
        f"{output_path}"
    )

    return output_path


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_rag_pipeline(
    llm_model: str = "mistral:7b",
):

    sys.path.append(
        str(
            PROJECT_ROOT / "src"
        )
    )

    from llm_chain import (
        ChainConfig,
        generate_answer,
    )
    from retriever import (
        RetrievalConfig,
        retrieve_documents,
    )

    retrieval_config = RetrievalConfig(
        top_k=6,
        max_distance=1.5,
    )

    chain_config = ChainConfig(
        llm_model=llm_model,
        request_timeout=300,
    )

    def rag_fn(
        question: str,
    ):

        documents, scores = (
            retrieve_documents(
                question,
                retrieval_config,
            )
        )

        answer = generate_answer(
            question,
            documents,
            chain_config,
        )

        chunks = [
            doc.page_content
            for doc in documents
        ]

        if not chunks:

            chunks = [
                "Aucun document récupéré."
            ]

        return answer, chunks

    return rag_fn


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Évaluation DeepEval du pipeline RAG "
            "pour l'assistant académique."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Fichier CSV ou Excel contenant "
            "les questions (colonne 'Question')"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de questions à évaluer",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "faithfulness",
            "answer_relevancy",
        ],
        help=(
            "Métriques à calculer. "
            "Options : faithfulness, "
            "answer_relevancy, "
            "contextual_relevancy, "
            "contextual_precision, "
            "contextual_recall, all"
        ),
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="mistral:7b",
        help=(
            "Modèle LLM pour la génération "
            "des réponses"
        ),
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="mistral:7b",
        help=(
            "Modèle LLM pour le juge "
            "DeepEval"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help=(
            "Seuil de réussite pour "
            "chaque métrique (défaut: 0.75)"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print(
        f"\n{'='*60}\n"
        f"  Évaluation DeepEval - "
        f"Assistant Académique\n"
        f"{'='*60}\n"
    )

    results = run_evaluation(
        input_path=args.input,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        metrics_list=args.metrics,
        limit=args.limit,
        threshold=args.threshold,
    )

    export_results(results)

    passed = sum(
        r["All_Passed"]
        for r in results
    )

    total = len(results)

    print(
        f"\n📊 Résumé : {passed}/{total} "
        f"tests réussis "
        f"({passed/total*100:.1f}%)"
    )

    print(
        f"\n{'='*60}\n"
        f"  Évaluation terminée.\n"
        f"{'='*60}"
    )


if __name__ == "__main__":

    main()
