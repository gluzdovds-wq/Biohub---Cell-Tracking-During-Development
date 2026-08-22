"""Run EXP026's complete source/window validation path without GPU training."""

from __future__ import annotations

import hashlib
import os
import runpy
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "95f704d07611d16fbfe413e3963cc21267392e11"
SOURCE_SHA256 = "060d1b8ad5abdd6823fee463650da32967199497dec473f68e0eba5d2f4612eb"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gluzdovds-wq/"
    "Biohub---Cell-Tracking-During-Development/"
    f"{SOURCE_COMMIT}/kaggle_notebooks/exp026_zebrahub_pretrain/pretrain_zebrahub.py"
)
DESTINATION = Path("/kaggle/working/pretrain_zebrahub_pinned.py")

request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "biohub-exp027/1.0"})
with urllib.request.urlopen(request, timeout=120) as response:
    source = response.read()
observed_sha256 = hashlib.sha256(source).hexdigest()
if observed_sha256 != SOURCE_SHA256:
    raise AssertionError(
        {"source_sha256": observed_sha256, "expected_source_sha256": SOURCE_SHA256}
    )
DESTINATION.write_bytes(source)
os.environ["BIOHUB_VALIDATE_ONLY"] = "1"
runpy.run_path(str(DESTINATION), run_name="__main__")
