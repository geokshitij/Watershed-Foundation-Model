#!/usr/bin/env bash
# update.sh
# Simple script to add, commit, and push all changes to main.

set -e  # stop if any command fails

git add .
git commit -m "updated" || echo "No changes to commit."
git push origin main

echo "✅ All changes pushed to origin/main."

