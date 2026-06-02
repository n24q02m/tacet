#!/usr/bin/env bash
# Fetch the official ProofWriter V2020.12.3 release (Tafjord et al., Findings
# of ACL 2021) used by the TACET deductive-reasoning experiments.  The archive
# is the publicly redistributable release hosted on the Allen Institute's
# aristo-data-public bucket; this script verifies its SHA-256 before extracting
# so a corrupt or partial download can never be mistaken for the real corpus.
#
# Usage:
#   bash scripts/fetch_proofwriter.sh [target_dir]
#
# After it completes:
#   $TARGET/proofwriter/proofwriter-dataset-V2020.12.3/CWA/depth-<d>/meta-<split>.jsonl
#
# which is exactly the layout the loader at tacet.data.load_proofwriter expects
# (its DATA_ROOT default is data/proofwriter/proofwriter-dataset-V2020.12.3).

set -euo pipefail

URL="https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
SHA256="bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"

TARGET="${1:-data}"
DEST="$TARGET/proofwriter"
ROOT="$DEST/proofwriter-dataset-V2020.12.3"
mkdir -p "$DEST"

if [[ -f "$ROOT/CWA/depth-2/meta-dev.jsonl" ]]; then
  echo "OK ProofWriter already present under $ROOT"
  exit 0
fi

ZIP="$DEST/proofwriter-dataset-V2020.12.3.zip"

sha_of() { sha256sum "$1" | awk '{print $1}'; }

# Download unless a complete, checksum-good archive is already on disk. A
# present-but-wrong file (typically a truncated download) is resumed with
# -C -; the SHA-256 check below is the real gate either way.
if [[ -f "$ZIP" && "$(sha_of "$ZIP")" == "$SHA256" ]]; then
  echo "OK archive already downloaded and verified"
else
  echo "Downloading ProofWriter V2020.12.3 (~204 MB) ..."
  curl -fSL -C - -o "$ZIP" "$URL"
fi

echo "Verifying SHA-256 ..."
got="$(sha_of "$ZIP")"
if [[ "$got" != "$SHA256" ]]; then
  echo "ERROR: checksum mismatch for $ZIP" >&2
  echo "  expected $SHA256" >&2
  echo "  got      $got" >&2
  echo "Delete the file and re-run; a partial download is the usual cause." >&2
  exit 1
fi
echo "OK checksum $SHA256"

echo "Extracting ..."
unzip -q -o "$ZIP" -d "$DEST"

if [[ ! -f "$ROOT/CWA/depth-2/meta-dev.jsonl" ]]; then
  echo "ERROR: expected $ROOT/CWA/depth-2/meta-dev.jsonl after extraction" >&2
  exit 1
fi

rm -f "$ZIP"
echo "OK ProofWriter ready under $ROOT"
