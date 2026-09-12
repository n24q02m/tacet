#!/usr/bin/env bash
# Enforce Conventional Commit prefixes: only feat: and fix: are allowed
# (chore(release)/chore(deps) are exempt: release bot generates the
# former; the latter is the mechanical uv.lock post-release sync, which
# must NOT trigger another release - the sync loop of 2026-09-12).
MSG=$(head -1 "$1")
if [[ "$MSG" =~ ^(feat|fix)(\(.+\))?:.+ ]] || [[ "$MSG" =~ ^chore\(release\):.+ ]] || [[ "$MSG" =~ ^chore\(deps\):.+ ]]; then
  exit 0
fi
echo "ERROR: Commit blocked. Only 'feat:' and 'fix:' prefixes allowed."
echo "Got: $MSG"
exit 1
