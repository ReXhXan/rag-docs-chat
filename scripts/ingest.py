"""CLI: ingest every supported file in a folder.  python -m scripts.ingest data/sample"""
import sys
from pathlib import Path

from app.core.loaders import SUPPORTED_EXTENSIONS
from app.main import get_pipeline

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
for f in sorted(folder.rglob("*")):
    if f.suffix.lower() in SUPPORTED_EXTENSIONS:
        print(f"{f.name}: {get_pipeline().ingest_file(f)} chunks")
