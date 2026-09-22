"""
Delete the Langfuse dataset and all its experiment runs, so we can start fresh.
Run: .venv/bin/python eval/reset_langfuse.py
"""
import os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from langfuse import Langfuse

DATASET_NAME = "NSW-Health-Services-Act-Golden-Set"

lf = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

confirm = input(f"Delete dataset '{DATASET_NAME}' and ALL its runs? (yes/no): ").strip()
if confirm != "yes":
    print("Aborted.")
    sys.exit(0)

try:
    lf.api.datasets.delete(dataset_name=DATASET_NAME)
    print(f"✓ Dataset '{DATASET_NAME}' deleted.")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\nTrying HTTP DELETE fallback...")
    import httpx
    resp = httpx.delete(
        f"https://cloud.langfuse.com/api/public/datasets/{DATASET_NAME}",
        auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]),
    )
    print(f"  Status: {resp.status_code}  Body: {resp.text[:200]}")
