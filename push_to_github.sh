#!/usr/bin/env bash
# Helper to add remote and push current branch to GitHub
# Usage: ./push_to_github.sh <git_repo_url>

set -euo pipefail
REPO_URL=${1:-}
if [ -z "$REPO_URL" ]; then
  echo "Usage: $0 <git_repo_url>"
  exit 1
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git branch -M main 2>/dev/null || true
git push -u origin main
