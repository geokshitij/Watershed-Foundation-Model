#!/usr/bin/env bash
# setup_github.sh
# Usage:
#   ./setup_github.sh <remote_url> [branch]
# Example:
#   ./setup_github.sh https://github.com/geokshitij/Watershed-Foundation-Model.git main
#
# This script:
#  - removes existing git metadata (.git directory)
#  - removes old .gitignore (so we recreate a clean one)
#  - writes a .gitignore that includes *.pth
#  - initializes git, commits all files
#  - sets remote to the provided URL (no token in URL)
#  - configures macOS keychain credential helper
#  - attempts to push to the remote branch you specify (default: main)
#
# NOTES:
#  - Don't embed tokens in the remote URL. When git prompts for credentials, paste your new token as the "password".
#  - If you prefer SSH, use an ssh remote URL (git@github.com:...).
#  - The script tries a second push with increased postBuffer if the first push fails due to large pack size.

set -euo pipefail

REMOTE_URL="${1:-https://github.com/geokshitij/Watershed-Foundation-Model.git}"
BRANCH="${2:-main}"
REPO_DIR="$(pwd)"

echo "Working in: $REPO_DIR"
echo "Remote URL: $REMOTE_URL"
echo "Branch: $BRANCH"
echo

# 0) Safety reminder
cat <<EOF
*** SECURITY REMINDER ***
If you have ever pasted a token into the terminal (a string starting with ghp_), treat it as compromised.
Go to GitHub -> Settings -> Developer settings -> Personal access tokens and revoke the exposed token now.
Create a new token (repo scope) or configure SSH keys instead.
EOF
echo

read -p "Press ENTER to continue (or Ctrl-C to abort)..." _ || true

# 1) Remove git metadata
if [ -d ".git" ]; then
  echo "Removing existing .git directory..."
  rm -rf .git
else
  echo "No .git directory found — continuing."
fi

# Also remove any git index files if present
if [ -f ".gitignore" ]; then
  echo "Removing existing .gitignore (we'll create a fresh one)..."
  rm -f .gitignore
fi

# 2) Create a new .gitignore
echo "Writing .gitignore..."
cat > .gitignore <<'GITIGNORE'
# Python / project
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
*.egg
dist/
build/

# Environment
env/
venv/
.venv/
.env
.env.local

# macOS
.DS_Store

# Editor
.vscode/
.idea/

# Logs
*.log

# Large model files (do NOT commit)
*.pth
*.pt
*.ckpt

# Archives
*.zip
*.tar.gz

# Others
.cache/
GITIGNORE

# 3) Initialize git, configure basic settings
echo "Initializing new git repository..."
git init -b "$BRANCH"

# 4) Add everything but respect .gitignore (we already created it)
echo "Adding files..."
git add .

# If there are extremely large files already staged, warn and abort before commit
# (this tries to detect files >200MB in the index)
LARGE_FILE_DETECTED=$(git ls-files -s | awk '{print $4}' | xargs -I{} bash -c 'test -f "{}" && du -k "{}" || true' 2>/dev/null | awk '$1 > 204800 {print $0}' || true)
if [ -n "$LARGE_FILE_DETECTED" ]; then
  echo
  echo "WARNING: One or more files larger than ~200MB appear present and staged. Commit aborted."
  echo "$LARGE_FILE_DETECTED"
  echo
  echo "Remove or untrack those large files (or use Git LFS) and re-run this script."
  exit 1
fi

echo "Committing..."
git commit -m "Initial commit (clean start) with .gitignore excluding large model files"

# 5) Set remote (no token embedded)
echo "Setting remote origin to: $REMOTE_URL"
git remote add origin "$REMOTE_URL" || { echo "Remote 'origin' already exists — replacing URL"; git remote set-url origin "$REMOTE_URL"; }

# 6) Configure macOS credential helper so token is stored in keychain on first use
# (If not on macOS, this will harmlessly fail — user can configure their own helper)
echo "Configuring credential helper (osxkeychain)..."
git config --global credential.helper osxkeychain || true

# 7) Push (first attempt)
echo
echo "Attempting to push to origin $BRANCH. You will be prompted for credentials if needed."
echo "Use your GitHub username as username and paste your NEW personal access token when asked for password."
echo
set +e
git push -u origin "$BRANCH"
PUSH_STATUS=$?
set -e

if [ $PUSH_STATUS -ne 0 ]; then
  echo
  echo "First push failed. Trying to increase git http.postBuffer and retry (may help with large pack sizes)..."
  git config --global http.postBuffer 524288000
  git config --global core.compression 0 || true

  echo "Retrying push..."
  set +e
  git push -u origin "$BRANCH"
  PUSH_STATUS=$?
  set -e

  if [ $PUSH_STATUS -ne 0 ]; then
    echo
    echo "Push still failed. Common causes:"
    echo " - authentication/permission issue (make sure your token has 'repo' scope or use SSH)"
    echo " - very large files committed (remove them; consider Git LFS)"
    echo " - network or proxy issues"
    echo
    echo "Diagnostics you can run:"
    echo "  git remote -v"
    echo "  git status --porcelain"
    echo "  git ls-files | grep '\\.pth$' || echo 'no .pth tracked'"
    echo "  GIT_TRACE_PACKET=1 GIT_TRACE=1 GIT_CURL_VERBOSE=1 git push -u origin $BRANCH"
    exit 2
  fi
fi

echo
echo "Success: repository pushed to remote (origin/$BRANCH)."
echo
echo "RECOMMENDATION: Now that push is done, revoke the exposed token at GitHub and create a new one,"
echo "or better: set up SSH keys and switch the remote to an ssh URL:"
echo
echo "  git remote set-url origin git@github.com:geokshitij/Watershed-Foundation-Model.git"
echo "  git push -u origin $BRANCH"
echo
echo "Script completed."
exit 0

