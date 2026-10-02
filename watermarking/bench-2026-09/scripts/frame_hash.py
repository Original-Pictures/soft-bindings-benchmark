"""Hash decoded frames of the stability fixtures, in the YUV planes and after PyAV's RGB
conversion, to separate "codec decode differs" from "colour conversion differs" from
"model differs" when two hosts disagree. Usage: python frame_hash.py > out.json"""

import hashlib
import json
import sys

import av
import numpy as np

from common import BENCH_WORK, host_info

out = {"host": host_info(), "av": av.__version__, "ffmpeg": av.library_versions.get("libswscale"), "files": {}}
for p in sorted((BENCH_WORK / "stability").glob("*.mp4")):
    yuv, rgb = hashlib.sha256(), hashlib.sha256()
    with av.open(str(p)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            for plane in f.to_ndarray(format="yuv420p"):
                yuv.update(np.ascontiguousarray(plane).tobytes())
            rgb.update(f.to_ndarray(format="rgb24").tobytes())
            if i >= 15:
                break
    out["files"][p.name] = {"yuv420p_16f": yuv.hexdigest()[:16], "rgb24_16f": rgb.hexdigest()[:16]}
json.dump(out, sys.stdout, indent=1, default=str)
