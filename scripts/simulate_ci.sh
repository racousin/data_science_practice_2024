#!/bin/bash

FETCH_ALL_MODULES=true

# Get the folder name from the current working directory as GITHUB_REPOSITORY_NAME
GITHUB_REPOSITORY_NAME=$(basename "$PWD")
export GITHUB_REPOSITORY_NAME

# Variables for options
AUTHOR=""
CHANGED_MODULES=""

# Function to process a specific author directory
process_author() {
    AUTHOR="$1"
    export AUTHOR
    echo "Processing author directory: ${AUTHOR}"

    # Simulate "Check and Add User to Students List"
    ./scripts/check-and-add-user.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION

    # Set CHANGED_MODULES if provided, otherwise check for all modules
    if [[ -z "$CHANGED_MODULES" ]]; then
        export CHANGED_MODULES=$(./scripts/check-changed-modules.sh $AUTHOR $FETCH_ALL_MODULES)
    fi

    # Simulate "Run Tests and Update Results"
    if [[ -z "$CHANGED_MODULES" ]]; then
        echo "No MODULEs changed for $AUTHOR. Skipping tests."
    else
        ./tests/run_tests_and_update_results.sh $AUTHOR $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION "$CHANGED_MODULES" $GITHUB_REPOSITORY_NAME
        ./scripts/compute_progress_percentage.sh $AUTHOR $GITHUB_REPOSITORY_NAME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
    fi
}

# Function to list all directories excluding the specified ones
list_authors() {
    for dir in $(ls -d */ | grep -vE 'venv/|scripts/|tests/|README.md'); do
        AUTHOR=$(basename "$dir")
        process_author "$AUTHOR"
    done
}

# Set permissions
chmod +x ./scripts/* ./tests/*/exercise*.sh ./tests/run_tests_and_update_results.sh

# Parse command line arguments
while getopts "a:m:" opt; do
    case ${opt} in
        a )
            AUTHOR=$OPTARG
            ;;
        m )
            CHANGED_MODULES=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        : )
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# If an author is provided, process that specific author
if [[ -n "$AUTHOR" ]]; then
    if [ -d "$AUTHOR" ]; then
        process_author "$AUTHOR"
    else
        echo "Error: Directory $AUTHOR does not exist."
        exit 1
    fi
else
    # Loop through all author directories if no specific author is provided
    list_authors
fi
