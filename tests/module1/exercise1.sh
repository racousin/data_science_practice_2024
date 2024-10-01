#!/bin/bash

# This script runs tests for module1 and expects a username as a parameter
# Usage: ./exercise1.sh <username>

USERNAME=$1
CURRENT_UTC_TIME=$2
MODULE_NUMBER="1"  # Since this script is specifically for module1, we can hardcode the module number.
RESULTS_DIR="./results"  # Directory to store results
RESULT_FILE="${RESULTS_DIR}/module${MODULE_NUMBER}_exercise1.json"  # File to store this exercise's results

mkdir -p $RESULTS_DIR  # Ensure results directory exists

echo "Starting tests for module1 for user $USERNAME..."

# Define the expected file path based on the username
FILE_PATH="${USERNAME}/module${MODULE_NUMBER}/user"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: File $FILE_PATH does not exist.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Check if the file contains the correct content
# We expect the format to be: 'username,something,something' (must contain 2 commas, but we're flexible on surname and name content)
FILE_CONTENT=$(cat "$FILE_PATH")
IFS=',' read -r -a content_array <<< "$FILE_CONTENT"

if [[ "${content_array[0]}" != "$USERNAME" ]] || [[ $(grep -o ',' <<< "$FILE_CONTENT" | wc -l) -ne 2 ]]; then
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: File content format is incorrect in $FILE_PATH. Received: '$FILE_CONTENT', Expected format: '${USERNAME},surname,name'\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# If the file exists and content is correct, output success message in JSON format
echo "{\"is_passed_test\": true, \"score\": \"100\", \"logs\": \"module${MODULE_NUMBER} tests passed successfully for ${USERNAME}.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
