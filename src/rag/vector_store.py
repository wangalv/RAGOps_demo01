"""
M7.1 — Vector Store

Wraps ChromaDB + SentenceTransformer into two operations:
  - add_chunks(): embed each chunk and persist to ChromaDB
  - search():     embed a query, return top-k similar chunks

Why SentenceTransformer (local) instead of OpenAI embeddings?
  - Free, no API key needed
  - all-MiniLM-L6-v2 is fast (~14ms/chunk on CPU) and good enough for English legal text
  - In production you'd swap this for a larger model or OpenAI text-embedding-3-small

Why ChromaDB?
  - Zero config, runs in-process, persists to disk
  - Easy to swap for Pinecone/Weaviate later — same interface
"""

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en-v1.5"   # 1024-dim, ~1.3GB download on first use


class VectorStore:
    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_dir)
        # get_or_create so we can re-run the test without re-embedding
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity, standard for text
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed each chunk and store in ChromaDB.

        ChromaDB requires:
          - ids:        unique string per document
          - embeddings: list of float vectors
          - documents:  raw text (stored for retrieval)
          - metadatas:  arbitrary dict per document
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"section_id": c["section_id"], "title": c["title"]} for c in chunks]

        print(f"  Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        # Upsert: safe to call multiple times — won't duplicate
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Stored {len(chunks)} chunks in ChromaDB collection '{self.collection.name}'")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed query and return top-k most similar chunks.

        Returns list of dicts with keys:
          - chunk_id, section_id, title, text, score (0=identical, 2=opposite for cosine)
        """
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id":   results["ids"][0][i],
                "section_id": results["metadatas"][0][i]["section_id"],
                "title":      results["metadatas"][0][i]["title"],
                "text":       results["documents"][0][i],
                "score":      round(1 - results["distances"][0][i], 4),  # convert to similarity
            })

        return chunks

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection — use when re-indexing from scratch."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_children(self, children: list[dict]) -> None:
        """Embed child chunks — same as add_chunks() but stores parent_id in metadata."""
        if not children:
            return

        texts     = [c["text"]      for c in children]
        ids       = [c["chunk_id"]  for c in children]
        metadatas = [{
            "parent_id":  c["parent_id"],
            "section_id": c["section_id"],
            "title":      c["title"],
        } for c in children]

        print(f"  Embedding {len(texts)} child chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        self.collection.upsert(ids=ids, embeddings=embeddings,
                               documents=texts, metadatas=metadatas)
        print(f"  Stored {len(children)} child chunks in '{self.collection.name}'")

    def search_children(self, query: str, top_k: int = 20) -> list[dict]:
        """Search child collection; each result includes parent_id for lookup."""
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id":  results["ids"][0][i],
                "parent_id": results["metadatas"][0][i]["parent_id"],
                "section_id":results["metadatas"][0][i]["section_id"],
                "title":     results["metadatas"][0][i]["title"],
                "text":      results["documents"][0][i],
                "score":     round(1 - results["distances"][0][i], 4),
            })
        return chunks

    def get_parents_by_ids(self, parent_store: "VectorStore",
                           parent_ids: list[str]) -> list[dict]:
        """Fetch full parent chunks from the parent collection by id.

        Deduplicates — multiple children may share the same parent.
        """
        unique_ids = list(dict.fromkeys(parent_ids))  # preserve order, deduplicate
        results = parent_store.collection.get(
            ids=unique_ids,
            include=["documents", "metadatas"],
        )
        parents = []
        for i, pid in enumerate(results["ids"]):
            parents.append({
                "chunk_id":  pid,
                "section_id":results["metadatas"][i]["section_id"],
                "title":     results["metadatas"][i]["title"],
                "text":      results["documents"][i],
            })
        return parents
