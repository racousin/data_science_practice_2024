#!/bin/bash

# Check if a commit hash is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <commit_hash>"
  exit 1
fi

# Set the commit hash to test
COMMIT_HASH=$1

# Navigate to the repository where the commit is located (assuming you're in a Git repo)
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Not inside a Git repository. Please run this script in a valid Git repository."
  exit 1
fi

# Check if the provided commit is a merge commit
if [ "$(git rev-parse --is-merge-commit $COMMIT_HASH)" = "1" ]; then
  echo "Merge commit detected for $COMMIT_HASH"

  # Get the second parent of the merge commit (usually the PR branch head)
  PR_AUTHOR=$(git log -1 --pretty=format:'%an' $COMMIT_HASH^2)
  PR_AUTHOR_EMAIL=$(git log -1 --pretty=format:'%ae' $COMMIT_HASH^2)
else
  echo "Regular commit detected for $COMMIT_HASH"

  # If not a merge commit, use the commit author
  PR_AUTHOR=$(git log -1 --pretty=format:'%an' $COMMIT_HASH)
  PR_AUTHOR_EMAIL=$(git log -1 --pretty=format:'%ae' $COMMIT_HASH)
fi

# Output the detected author information
echo "Detected Author: $PR_AUTHOR"
echo "Detected Author Email: $PR_AUTHOR_EMAIL"

# Check if the author was successfully determined
if [ -n "$PR_AUTHOR" ] && [ "$PR_AUTHOR" != "null" ]; then
  echo "Author successfully determined: $PR_AUTHOR"
else
  echo "Failed to determine author for commit $COMMIT_HASH"
  exit 1
fi
