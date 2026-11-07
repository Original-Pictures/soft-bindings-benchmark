"""Run the paper's image and watermark-combination stages with the ISCC Semantic-Code methods added.

The paper ran ISCC-SCI-256 in its pilot only (fingerprints_image.SMOKE_ONLY_METHODS), so its image track does not
know the method. This wrapper adds it to IMAGE_METHODS, with its own adapter, together with two methods made from
stored descriptors:

  ISCC-SCI-64        the first 64 bits of ISCC-SCI-256 (iscc-sci cuts its 256-bit digest to shorter lengths)
  ISCC-Combined-512  ISCC-SCI-256 followed by ISCC-Image-256, compared as one 512-bit code (the usual order of
                     ISCC units, from the more abstract to the more concrete)

bench_image.py and bench_wmcombo.py run unchanged. Registration happens at import, so the spawned worker processes, which
import this module again, know the methods too.

    python bench.py image extract|search|metrics [args]   bench_image.py with the added methods
    python bench.py wmcombo render|decode|analyze [args]  bench_wmcombo.py with the added methods
    python bench.py derive                                write the stored-descriptor methods for every set
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fingerprints_image as fi  # noqa: E402
from common import DESC  # noqa: E402

# name -> (source method, leading bits) of each part, concatenated in this order
DERIVED = {
    "ISCC-SCI-64": (("ISCC-SCI-256", 64),),
    "ISCC-Combined-512": (("ISCC-SCI-256", 256), ("ISCC-Image-256", 256)),
}
SCI_WEIGHTS = "none stated; ISC21-derived (DISC21 CC BY-NC)"  # as fingerprints_image.ISCCSemantic records it


def derived_method(name: str) -> fi.Fingerprinter:
    """Method record of a stored-descriptor method; it is never extracted, only searched and scored."""
    parts = DERIVED[name]
    semantic = any(src.startswith("ISCC-SCI") for src, _ in parts)
    return fi.Fingerprinter(name=name, family="ISCC", metric="hamming", dim=sum(b for _, b in parts),
                            tier="R" if semantic else "P", code_licence="Apache-2.0",
                            weights_licence=SCI_WEIGHTS if semantic else "n/a",
                            params={"parts": [f"{src}[:{b}]" for src, b in parts]})


def register() -> None:
    """Add the Semantic-Code adapter and the stored-descriptor methods to the paper's method table."""
    fi.IMAGE_METHODS.update(fi.SMOKE_ONLY_METHODS)
    fi.IMAGE_METHODS.update({n: functools.partial(derived_method, n) for n in DERIVED})


def combine(arrays: list[np.ndarray], bits: list[int]) -> np.ndarray:
    """The leading bits of each 0/1 descriptor array, concatenated per row."""
    return np.concatenate([a[:, :b] for a, b in zip(arrays, bits)], axis=1)


def derive(desc: Path = DESC) -> None:
    """Write each stored-descriptor method for every descriptor set that holds all of its parts."""
    for ids in sorted(desc.glob("*/ids.json")):
        d = ids.parent
        for name, parts in DERIVED.items():
            files = [d / f"{src}.npy" for src, _ in parts]
            if all(f.exists() for f in files):
                np.save(d / f"{name}.npy", combine([np.load(f) for f in files], [b for _, b in parts]))
                print(f"{d.name}: {name}", flush=True)


register()

if __name__ == "__main__":
    stage, args = sys.argv[1], sys.argv[2:]
    if stage == "derive":
        derive()
    else:
        import bench_image
        import bench_wmcombo

        sys.argv = [f"bench_{stage}.py", *args]
        {"image": bench_image, "wmcombo": bench_wmcombo}[stage].main()
