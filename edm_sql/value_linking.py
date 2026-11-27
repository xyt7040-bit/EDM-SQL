"""Local embedding retrieval for the value-based schema-linking branch."""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha1
from itertools import groupby
from pathlib import Path
from uuid import uuid4

import numpy as np


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BATCH_SIZE = 64
VALUES_PER_COLUMN = 5
INDEX_VERSION = 1
UUID_VALUE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
NUMERIC_VALUE = re.compile(r"[0-9]+\.?[0-9]*")


def add_index_arguments(parser):
    parser.add_argument("--embedding-model",
                        default=os.environ.get("EDM_SQL_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model ID or local directory used to build and query the index")
    parser.add_argument("--embedding-device", default=os.environ.get("EDM_SQL_EMBEDDING_DEVICE") or None,
                        help="Optional embedding device, e.g. cpu or cuda:0")
    parser.add_argument("--value-index-dir", type=Path, default=Path(".value_index"),
                        help="Directory for the offline value indexes")


class LocalEmbedding:
    def __init__(self, model_name: str, device: str | None = None):
        model_path = Path(model_name).expanduser()
        self.model_name = str(model_path.resolve()) if model_path.is_dir() else model_name
        self.device = device
        self.model = None

    def __call__(self, texts: list[str]):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
        # Use the same Qwen query prompt for both sides as the experimental linker.
        return self.model.encode(
            texts, prompt_name="query", batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )


class ValueIndex:
    def __init__(self, root: Path, model_name: str):
        self.root = root
        self.model_name = model_name

    def index_path(self, db) -> Path:
        return self.root / db.path.stem / "local_index"

    def metadata(self, db) -> dict:
        stat = db.path.stat()
        return {"model": self.model_name, "database": os.path.normcase(str(db.path)),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def require(self, db) -> Path:
        path = self.index_path(db)
        hint = f"Run build_value_index.py with the same input, model and index directory ({self.root})."
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No offline value index for {db.path}. {hint}")
        manifest = read_manifest(path)
        if manifest.get("edm_sql") == self.metadata(db):
            return path
        if "edm_sql" not in manifest:
            raise ValueError(f"Index at {path} lacks EDM-SQL metadata. "
                             "Use build_value_index.py with a new --value-index-dir.")
        raise ValueError(f"Value index for {db.path} is incompatible or stale. {hint}")

    def build(self, db, embed) -> Path:
        metadata = self.metadata(db)
        path = self.index_path(db).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            previous = read_manifest(path).get("edm_sql")
            if not isinstance(previous, dict) or previous.get("database") != metadata["database"]:
                raise ValueError(f"Refusing to overwrite an unrecognized or different database index: {path}. "
                                 "Use a new --value-index-dir.")
        staging = path.parent / f".build-{uuid4().hex}"
        backup = path.parent / f".backup-{uuid4().hex}"
        (staging / "columns").mkdir(parents=True)
        try:
            entries = []
            batches = db.text_value_batches(EMBEDDING_BATCH_SIZE)
            for (table, column), group in groupby(batches, key=lambda batch: batch[0]):
                values = [value for _, batch in group for value in batch]
                if all(UUID_VALUE.fullmatch(value) for value in values) or all(
                    NUMERIC_VALUE.fullmatch(value) for value in values
                ):
                    continue
                entries.append(write_column(staging, table, column, values, embed))
            (staging / "manifest.json").write_text(json.dumps({
                "version": INDEX_VERSION, "metric": "cosine", "edm_sql": metadata,
                "columns": sorted(entries, key=lambda entry: entry["column_key"]),
            }, indent=2), encoding="utf-8")
            if self.metadata(db) != metadata:
                raise ValueError(f"Database changed during index construction: {db.path}")
            if path.exists():
                path.rename(backup)
            try:
                staging.rename(path)
            except OSError:
                if backup.exists() and not path.exists():
                    backup.rename(path)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return path


def read_manifest(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("version") != INDEX_VERSION
            or manifest.get("metric") != "cosine" or not isinstance(manifest.get("columns"), list)):
        raise ValueError(f"Unsupported local-index manifest: {path}")
    return manifest


def write_column(path, table, column, values, embed):
    table_name, column_name = table.lower(), column.lower()
    key = f"{table_name}\t{column_name}"
    tag = sha1(key.encode("utf-8")).hexdigest()
    documents = f"columns/{tag}.documents.json"
    embeddings = f"columns/{tag}.embeddings.npy"
    dimension = 0
    # Stream batches into one NumPy array file per column without keeping its vectors in RAM.
    with (path / embeddings).open("wb") as stream:
        for start in range(0, len(values), EMBEDDING_BATCH_SIZE):
            batch = values[start:start + EMBEDDING_BATCH_SIZE]
            vectors = np.asarray(embed(batch), dtype="<f4")
            if (vectors.ndim != 2 or vectors.shape[0] != len(batch) or not vectors.shape[1]
                    or (dimension and dimension != vectors.shape[1])):
                raise ValueError("Unexpected shape of value embeddings")
            if not dimension:
                dimension = vectors.shape[1]
                np.lib.format.write_array_header_2_0(stream, {
                    "descr": "<f4", "fortran_order": False, "shape": (len(values), dimension),
                })
            stream.write(vectors.tobytes())
    (path / documents).write_text(json.dumps(values), encoding="utf-8")
    return {"table_name": table_name, "column_name": column_name, "column_key": key,
            "original_table_name": table, "original_column_name": column,
            "documents_file": documents, "embeddings_file": embeddings,
            "count": len(values), "embedding_dim": dimension}


def index_file(root: Path, name: str) -> Path:
    path = (root / name.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Index file is outside its local index: {name}")
    return path


def indexed_batches(path: Path):
    path = path.resolve()
    for entry in read_manifest(path)["columns"]:
        table = entry.get("original_table_name", entry["table_name"])
        column = entry.get("original_column_name", entry["column_name"])
        documents = json.loads(index_file(path, entry["documents_file"]).read_text(encoding="utf-8"))
        vectors = np.load(index_file(path, entry["embeddings_file"]), mmap_mode="r", allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape != (len(documents), entry["embedding_dim"]):
            raise ValueError(f"Invalid value-index vectors for {table}.{column}")
        for start in range(0, len(documents), 4096):
            yield (table, column), documents[start:start + 4096], vectors[start:start + 4096]


def retrieve_values(index_path: Path, keywords, embed) -> list[dict]:
    if not keywords:
        return []
    queries = embed(keywords)
    links = []
    batches = indexed_batches(index_path)
    for (table, column), column_batches in groupby(batches, key=lambda batch: batch[0]):
        best = []
        for _, values, vectors in column_batches:
            if vectors.shape[1] != queries.shape[1]:
                raise ValueError("Value-index dimensions do not match the query embedding model")
            scores = (vectors @ queries.T).max(axis=1)
            ranked = best + [(-float(score), value) for score, value in zip(scores, values)]
            best = sorted(ranked)[:VALUES_PER_COLUMN]
        links.extend({"table": table, "column": column, "value": value} for _, value in best)
    return links
