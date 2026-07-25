"""Cut a new version of the Zenodo deposit from the built paper.

The record is managed through the LEGACY Deposit API and its metadata is in the
legacy shape. The RDM endpoint (``/api/records/{id}/draft``) must not be used:
its read form is not its write form, so a read-modify-write round trip silently
drops ``creators``, ``resource_type`` and ``license`` and the publish then fails.
This script therefore reads the metadata of the latest published version verbatim
and overrides only what a new version changes.

Publishing is irreversible. The script stops at the draft unless ``--publish`` is
passed, and prints the draft URL so a human can look at it first.

Usage::

    python scripts/zenodo_release.py --version v7 --pdf paper/build/main.pdf
    python scripts/zenodo_release.py --version v7 --pdf ... --publish

Reads ``ZENODO_TOKEN`` from the environment. The token needs the
``deposit:write`` and ``deposit:actions`` scopes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC
from pathlib import Path
from typing import Any

API = "https://zenodo.org/api"
CONCEPT_RECORD = "20621240"  # concept DOI 10.5281/zenodo.20621240

# Fields that belong to the version being copied FROM, never to the new draft.
DROP = ("doi", "prereserve_doi")


class ZenodoError(RuntimeError):
    pass


def _call(method: str, url: str, token: str, body: Any = None, raw: bytes | None = None) -> Any:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if raw is not None:
        # urllib defaults a bytes body to application/x-www-form-urlencoded,
        # which the files API rejects with a 415.
        req.add_header("Content-Type", "application/octet-stream")
    elif body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read()
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:800]
        raise ZenodoError(f"{method} {url} -> {e.code}\n{detail}") from e


def latest_published_id(token: str) -> str:
    """The id of the currently published latest version of the concept record."""
    rec = _call("GET", f"{API}/records/{CONCEPT_RECORD}/versions/latest", token)
    return str(rec["id"])


def new_version_draft(token: str, published_id: str) -> dict:
    """Open a new-version draft. The action returns the ORIGINAL record, not the draft."""
    original = _call("POST", f"{API}/deposit/depositions/{published_id}/actions/newversion", token)
    draft_url = original["links"]["latest_draft"]
    return _call("GET", draft_url, token)


def build_metadata(
    published: dict, version: str, publication_date: str, description: str | None
) -> dict:
    """Legacy metadata of the published version, with only the new-version fields moved."""
    meta = dict(published["metadata"])
    for key in DROP:
        meta.pop(key, None)
    meta["version"] = version
    meta["publication_date"] = publication_date
    if description:
        meta["description"] = description
    return meta


def deposit_filename(draft: dict) -> str | None:
    """The name a file already carries on the draft, if Zenodo inherited one."""
    files = draft.get("files") or []
    return files[0].get("filename") if files else None


def replace_file(token: str, draft: dict, pdf: Path, remote_name: str) -> None:
    """Write the built PDF into the draft's bucket under ``remote_name``.

    A new-version draft inherits the previous version's file, and deleting an
    inherited file answers 500. Writing the SAME key into the bucket replaces
    the object instead, so the inherited copy never has to be removed. Anything
    left over under a different name is deleted, which is a file this run put
    there and can therefore be removed normally.
    """
    bucket = draft["links"]["bucket"]
    _call("PUT", f"{bucket}/{remote_name}", token, raw=pdf.read_bytes())
    for existing in _call("GET", f"{API}/deposit/depositions/{draft['id']}/files", token) or []:
        if existing.get("filename") != remote_name:
            _call(
                "DELETE", f"{API}/deposit/depositions/{draft['id']}/files/{existing['id']}", token
            )


def verify_upload(token: str, draft_id: str, pdf: Path, remote_name: str) -> None:
    import hashlib

    local = hashlib.md5(pdf.read_bytes()).hexdigest()
    files = _call("GET", f"{API}/deposit/depositions/{draft_id}/files", token) or []
    if len(files) != 1:
        raise ZenodoError(
            f"expected exactly one file on the draft, found {len(files)}: "
            + ", ".join(f.get("filename", "?") for f in files)
        )
    if files[0].get("filename") != remote_name:
        raise ZenodoError(f"draft holds {files[0].get('filename')!r}, expected {remote_name!r}")
    remote = files[0].get("checksum", "").removeprefix("md5:")
    if remote != local:
        raise ZenodoError(f"checksum mismatch: local {local} != remote {remote}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True, help="version label, e.g. v7")
    ap.add_argument("--pdf", required=True, type=Path, help="the built paper")
    ap.add_argument("--date", default=None, help="publication_date (default: today, UTC)")
    ap.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help="HTML description; omit to keep the previous version's",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="publish the draft. IRREVERSIBLE; without it the script stops at the draft",
    )
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    if not args.pdf.is_file() or args.pdf.stat().st_size == 0:
        print(f"{args.pdf} is missing or empty", file=sys.stderr)
        return 2

    date = args.date
    if not date:
        from datetime import datetime

        date = datetime.now(UTC).strftime("%Y-%m-%d")

    description = None
    if args.description_file:
        description = args.description_file.read_text(encoding="utf-8").strip()

    published_id = latest_published_id(token)
    published = _call("GET", f"{API}/deposit/depositions/{published_id}", token)
    print(f"latest published version: {published_id} ({published['metadata'].get('version')})")

    draft = new_version_draft(token, published_id)
    draft_id = str(draft["id"])
    print(f"new-version draft: {draft_id}")

    metadata = build_metadata(published, args.version, date, description)
    _call("PUT", f"{API}/deposit/depositions/{draft_id}", token, body={"metadata": metadata})
    print(f"metadata set: version={args.version} date={date}")

    # Keep the deposit's established filename across versions: the bucket key is
    # what makes a write replace the inherited file rather than sit beside it.
    remote_name = deposit_filename(draft) or deposit_filename(published) or args.pdf.name
    replace_file(token, draft, args.pdf, remote_name)
    verify_upload(token, draft_id, args.pdf, remote_name)
    print(f"uploaded as {remote_name} ({args.pdf.stat().st_size} bytes), checksum verified")

    if not args.publish:
        print("\nDRAFT ONLY. Review it, then re-run with --publish:")
        print(f"  https://zenodo.org/uploads/{draft_id}")
        return 0

    result = _call("POST", f"{API}/deposit/depositions/{draft_id}/actions/publish", token)
    print(f"published: DOI {result.get('doi')} (concept {result.get('conceptdoi')})")

    latest = _call("GET", f"{API}/records/{CONCEPT_RECORD}/versions/latest", token)
    if str(latest["id"]) != draft_id:
        raise ZenodoError(f"concept record resolves to {latest['id']}, expected {draft_id}")
    print(f"concept DOI resolves to the new version ({draft_id})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ZenodoError as e:
        print(f"\nZenodo API error:\n{e}", file=sys.stderr)
        raise SystemExit(1) from e
