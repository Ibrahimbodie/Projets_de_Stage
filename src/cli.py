from __future__ import annotations

import argparse
from typing import Iterable

from llm_chain import ChainConfig, format_sources, generate_answer
from retriever import RetrievalConfig, retrieve_documents


def _print_sources(sources: Iterable[str]) -> None:
    source_list = list(sources)
    if not source_list:
        return
    print("\nSources:")
    print(", ".join(source_list))


def answer_once(
    question: str,
    top_k: int,
    min_similarity: float,
    llm_model: str,
) -> None:
    retrieval_config = RetrievalConfig(top_k=top_k, min_similarity=min_similarity)
    documents, _ = retrieve_documents(question, retrieval_config)
    answer = generate_answer(question, documents, ChainConfig(llm_model=llm_model))
    sources = format_sources(documents)

    print("\nRéponse:")
    print(answer)
    _print_sources(sources)


def run_interactive(top_k: int, min_similarity: float, llm_model: str) -> None:
    print("Assistant académique Master SIM (terminal)")
    print("Commande: 'exit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()
        if not question:
            print("Merci de saisir une question.")
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Fin de session.")
            break

        try:
            answer_once(question, top_k, min_similarity, llm_model)
            print()
        except Exception as exc:
            print(f"Erreur inattendue: {exc}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistant Master SIM en terminal.")
    parser.add_argument(
        "--question",
        type=str,
        help="Question unique à exécuter. Si absent, la CLI démarre en mode interactif.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Nombre max de passages récupérés.")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.3,
        help="Seuil de similarité normalisé entre 0 et 1.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3.5:27b",
        help="Nom du modèle Ollama pour la génération.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.question:
            question = args.question.strip()
            if not question:
                print("Erreur: la question est vide.")
                return
            answer_once(question, args.top_k, args.min_similarity, args.llm_model)
            return
        run_interactive(args.top_k, args.min_similarity, args.llm_model)
    except FileNotFoundError as exc:
        print(str(exc))
    except ConnectionError:
        print(
            "Connexion Ollama impossible. Vérifie que Ollama est démarré "
            "et que les modèles nécessaires sont disponibles."
        )
    except Exception as exc:
        print(f"Erreur inattendue: {exc}")


if __name__ == "__main__":
    main()
