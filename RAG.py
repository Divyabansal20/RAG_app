import os
import uuid
from collections import deque

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from endee import Endee, Precision


class RAGApp:
    def __init__(self):
        # Models
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.generator = AutoModelForCausalLM.from_pretrained(
            "models",
            model_file="mistral.gguf",
            model_type="mistral"
        )

        # Endee setup
        self.db = Endee()
        self.db.set_base_url("http://localhost:8080/api/v1")

        self.index_name = "rag_index"
        self.dim = 384
        self._setup_index()
        self.index = self.db.get_index(self.index_name)

        # Short conversation memory
        self.chat_history = deque(maxlen=3)

    # -------------------------
    # Index setup
    # -------------------------
    def _setup_index(self):
        try:
            self.db.create_index(
                name=self.index_name,
                dimension=self.dim,
                space_type="cosine",
                precision=Precision.INT8
            )
        except Exception:
            pass

    # -------------------------
    # Load & ingest PDFs
    # -------------------------
    def load_documents(self, folder="data/docs"):
        text_blob = []

        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                reader = PdfReader(os.path.join(folder, file))
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text_blob.append(content)

        full_text = "\n".join(text_blob)
        segments = self._chunkify(full_text)

        vectors = self.encoder.encode(segments)
        records = []

        for vec, seg in zip(vectors, segments):
            records.append({
                "id": str(uuid.uuid4()),
                "vector": vec.tolist(),
                "meta": {"text": seg}
            })

        self.index.upsert(records)

    # -------------------------
    # Semantic retrieval
    # -------------------------
    def find_context(self, question, top_k=2):
        q_vec = self.encoder.encode([question])[0]
        hits = self.index.query(
            vector=q_vec.tolist(),
            top_k=top_k
        )
        return "\n".join(hit["meta"]["text"] for hit in hits)

    # -------------------------
    # RAG answer generation
    # -------------------------
    def answer(self, question):
        context = self.find_context(question)

        memory_block = "\n".join(
            f"Q: {m['question']}\nA: {m['answer']}"
            for m in self.chat_history
        )

        prompt = f"""
Use only the information given in the context to answer.
If the answer cannot be inferred, reply with:
"I don't know based on the document."

Chat History:
{memory_block}

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.generator(prompt, max_new_tokens=120)

        self.chat_history.append({
            "question": question,
            "answer": response
        })

        return response

    # -------------------------
    # Text chunking helper
    # -------------------------
    def _chunkify(self, text, length=400, overlap=50):
        chunks = []
        pos = 0
        while pos < len(text):
            chunks.append(text[pos:pos + length])
            pos += length - overlap
        return chunks


if __name__ == "__main__":
    app = RAGApp()

    print("Commands:")
    print("  ingest  -> load PDFs into vector database")
    print("  exit    -> quit")
    print("  or type any question to query\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "ingest":
            app.load_documents()
            print("Ingestion completed.")
        elif user_input.lower() in ("exit", "quit"):
            break
        else:
            reply = app.answer(user_input)
            print("\n", reply)