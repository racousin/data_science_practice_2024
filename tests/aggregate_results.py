import os
import json
import boto3
import sys


def download_from_s3(
    bucket, key, aws_access_key_id, aws_secret_access_key, region_name, download_path
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    s3 = session.client("s3")
    s3.download_file(bucket, key, download_path)


def safe_load_json(file_path):
    """
    Safely load JSON file with error handling and debugging information.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"\nError decoding JSON in file: {file_path}")
                print(f"Error details: {str(e)}")
                print("\nFile content:")
                print("-" * 50)
                print(content)
                print("-" * 50)
                print(f"\nFile size: {os.path.getsize(file_path)} bytes")
                print(f"Is file empty? {os.path.getsize(file_path) == 0}")
                if content.strip() == '':
                    print("File is empty or contains only whitespace")
                raise
    except FileNotFoundError:
        print(f"\nFile not found: {file_path}")
        raise


def aggregate_results(
    results_dir,
    final_json_path,
    aws_access_key_id,
    aws_secret_access_key,
    region_name,
    bucket,
    key,
):
    # Download the existing results JSON from S3
    download_from_s3(
        bucket,
        key,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
        final_json_path,
    )
    print(f"download done to {final_json_path}")

    # Load the existing results with error handling
    all_results = safe_load_json(final_json_path)

    # Merge local results into the existing JSON
    for filename in os.listdir(results_dir):
        print(filename)
        print("exercise" in filename)
        if "exercise" in filename:
            module, exercise = filename.replace(".json", "").split("_")
            print(f"extract module: {module}, exercise: {exercise}")
            file_path = os.path.join(results_dir, filename)
            result = safe_load_json(file_path)
            if module in all_results and exercise in all_results[module]:
                all_results[module][exercise] = result
                print(f"content: {result}")

    # Write the merged results back to the JSON file
    with open(final_json_path, "w") as file:
        json.dump(all_results, file)
    print(f"write back json locally {all_results}")
    return final_json_path


def upload_to_s3(
    file_path, bucket, key, aws_access_key_id, aws_secret_access_key, region_name
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    s3 = session.client("s3")
    s3.upload_file(file_path, bucket, key)
    print(f"write back json remotly from path {file_path} to {key}")


if __name__ == "__main__":
    results_dir = sys.argv[1]
    final_json_path = sys.argv[2]
    bucket = sys.argv[3]
    key = sys.argv[4]
    aws_access_key_id = sys.argv[5]
    aws_secret_access_key = sys.argv[6]
    region_name = sys.argv[7]

    final_path = aggregate_results(
        results_dir,
        final_json_path,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
        bucket,
        key,
    )
    upload_to_s3(
        final_path, bucket, key, aws_access_key_id, aws_secret_access_key, region_name
    )