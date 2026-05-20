from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".docx",
]


def load_user_document(file_path: Path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        loader = PyPDFLoader(str(file_path))

    elif suffix == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    elif suffix == ".docx":

        loader = Docx2txtLoader(str(file_path))

    else:

        raise ValueError(
            f"Format non supporté : {suffix}"
        )

    return loader.load()