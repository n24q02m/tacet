#!/usr/bin/env bash
# Pull the public KGC benchmark datasets used by the TACET experiment
# scripts.  All sources are publicly redistributable academic mirrors.
#
# Usage:
#   bash scripts/fetch_kgc_data.sh [target_dir]
#
# After it completes:
#   $TARGET/FB15k-237/{train,valid,test}.txt
#   $TARGET/WN18RR/{train,valid,test}.txt
#   $TARGET/MetaQA/...                       (manual: see notes)

set -euo pipefail

TARGET="${1:-data}"
mkdir -p "$TARGET"

clone() {
  local dest="$1" url="$2"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" pull --quiet
  else
    git clone --depth=1 "$url" "$dest"
  fi
}

# ConvE ships FB15k-237 / WN18RR / YAGO3-10 tarballs in its repo.
TMP="$(mktemp -d)"
trap "rm -rf $TMP" EXIT
clone "$TMP/conve" "https://github.com/TimDettmers/ConvE.git"

for dataset in FB15k-237 WN18RR YAGO3-10; do
  if [[ -f "$TMP/conve/${dataset}.tar.gz" ]]; then
    mkdir -p "$TARGET/${dataset}"
    tar -xzf "$TMP/conve/${dataset}.tar.gz" -C "$TARGET/${dataset}"
    echo "OK ${dataset}: $(wc -l "$TARGET/${dataset}"/*.txt | tail -1 | awk '{print $1}') triples"
  else
    echo "skip ${dataset}: tarball not found in ConvE"
  fi
done

cat <<'NOTE'

MetaQA is distributed via Google Drive and is not directly downloadable
from this script.  Get it once with:

  https://github.com/yuyuz/MetaQA  (clone the repo, then follow its README
                                    to download the data folder from Drive)

Once unpacked under "$TARGET/MetaQA" with the expected layout
(kb.txt + 1-hop/qa_*.txt + 2-hop/qa_*.txt + 3-hop/qa_*.txt) the
experiments/run_metaqa.py runner picks it up automatically.
NOTE
