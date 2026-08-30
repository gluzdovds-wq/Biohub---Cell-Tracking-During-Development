"""Freeze exact reviewed public notebooks for the August 30 batch."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs/research/frontier_20260830"
FROZEN_ROOT = ROOT / "research/frozen_sources_20260830"
CASES = [
    (
        "grafael__biohub-ct-0940-ema__1/biohub-ct-0940-ema.ipynb",
        "exp088_biohub-ct-0940-ema_v1.ipynb",
        "4f0fb3aff772f4aaa0b687477e255c05211412a458d9db085935a7f37f073f33",
    ),
    (
        "notoverkil__biohub-base0931-edgebar040__1/biohub-base0931-edgebar040.ipynb",
        "exp090_biohub-base0931-edgebar040_v1.ipynb",
        "9b98f72c6763a49d260247ce5abd2de8f455cbc291896a62e8e89edb735e4fbc",
    ),
    (
        "salemali7__biohub-cell-tracking-92-6__7/biohub-cell-tracking-92-6.ipynb",
        "exp091_biohub-cell-tracking-92-6_v7.ipynb",
        "eb45f2bd542a063683a79f5f7c6b102a78033111962e6fb09ebc1f750c8dbb53",
    ),
    (
        "muhanqiu__biohub-final-submission-our-weights__1/biohub-final-submission-our-weights.ipynb",
        "exp092_parent_biohub-final-submission-our-weights_v1.ipynb",
        "fae5d1b15bd1c3876d2334f50c2555403bf36904d10b70fb1a1b24f48078a596",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    for relative_source, frozen_name, expected_hash in CASES:
        source = SOURCE_ROOT / relative_source
        if sha256(source) != expected_hash:
            raise RuntimeError(f"Reviewed source drift: {source}")
        destination = FROZEN_ROOT / frozen_name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Refusing to replace different snapshot: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"Frozen snapshot hash mismatch: {destination}")
        print(f"FROZEN {frozen_name} {expected_hash}")


if __name__ == "__main__":
    main()
