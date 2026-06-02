#!/usr/bin/env bash
# Enforce Conventional Commit prefixes: only feat: and fix: are allowed
# (chore(release): is exempt because python-semantic-release generates it).
MSG=$(head -1 "$1")
if [[ "$MSG" =~ ^(feat|fix)(\(.+\))?:.+ ]] || [[ "$MSG" =~ ^chore\(release\):.+ ]]; then
  exit 0
fi
echo "ERROR: Commit blocked. Only 'feat:' and 'fix:' prefixes allowed."
echo "Got: $MSG"
exit 1
