#!/bin/bash

FETCH_ALL_MODULES=true

# Simulate "Identify Commit Author"
AUTHOR=$(git log -1 --pretty=format:'%an')
export AUTHOR

# Simulate "Setup AWS CLI"
export GITHUB_REPOSITORY_NAME="data_science_practice"

# Set permissions
chmod +x ./scripts/* ./tests/*/exercise*.sh ./tests/run_tests_and_update_results.sh

# Simulate "Check and Add User to Students List"
./scripts/check-and-add-user.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION

# Simulate "Determine Changed Modules"
export CHANGED_MODULES=$(./scripts/check-changed-modules.sh $AUTHOR $FETCH_ALL_MODULES)

# Simulate "Run Tests and Update Results"
if [[ -z "$CHANGED_MODULES" ]]; then
  echo "No MODULEs changed. Skipping tests."
else
  ./tests/run_tests_and_update_results.sh $AUTHOR $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION "$CHANGED_MODULES" $GITHUB_REPOSITORY_NAME
  ./scripts/compute_progress_percentage.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
fi
