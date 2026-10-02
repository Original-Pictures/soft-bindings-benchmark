"""Try to fetch InvisMark weights from the README's public OneDrive folder, anonymously.

InvisMark's weights carry NO licence statement, so this is a reference-only row at
best. If the share cannot be listed without a login, the model is recorded as
"unavailable" in the report; we do not work around authentication.
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.request

from common import WEIGHTS, sha256_file

SHARE = "https://1drv.ms/f/c/7882afab383c8474/Ei_Lasu5CrpHsrNIkYRLenYBmx662VSAovq5hD8r-NsB5A"


def api(path: str) -> dict:
    tok = "u!" + base64.urlsafe_b64encode(SHARE.encode()).decode().rstrip("=")
    req = urllib.request.Request(f"https://api.onedrive.com/v1.0/shares/{tok}/{path}", headers={"User-Agent": "op-wm-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> None:
    try:
        root = api("driveItem?$expand=children")
    except Exception as exc:
        print(f"UNAVAILABLE: anonymous listing refused ({exc})")
        sys.exit(2)
    out = WEIGHTS / "invismark"
    out.mkdir(parents=True, exist_ok=True)
    for child in root.get("children", []):
        print(child.get("name"), child.get("size"), "folder" if "folder" in child else "")
        dl = child.get("@content.downloadUrl")
        if dl and child.get("size", 0) < 2_000_000_000:
            dest = out / child["name"]
            urllib.request.urlretrieve(dl, dest)
            print("  sha256", sha256_file(dest))


if __name__ == "__main__":
    main()
