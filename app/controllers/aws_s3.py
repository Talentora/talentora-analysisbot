import boto3
from botocore.exceptions import ClientError

def fetch_from_s3(bucket: str, file_path: str, local_path: str) -> bool:
    """Download an S3 object to local_path. Returns True on success."""
    s3 = boto3.client("s3")        # picks up credentials & region automatically
    try:
        s3.download_file(bucket, file_path, local_path)
        print(f"✓ Downloaded s3://{bucket}/{file_path} → {local_path}")
        return True
    except ClientError as err:
        code = err.response["Error"]["Code"]
        if code == "404":
            print("Object does not exist!")
        else:
            print("S3 error:", code, err.response["Error"]["Message"])
        return False



if __name__ == "__main__":
    # Usage
    fetch_from_s3(
        bucket="talentorarecordings",
        key="af4be366-5f9a-46a1-a48c-0788363168c0/6590b339-db89-415a-bc62-b3599af6bc48/interview_recording.mp4",
        local_path="./interview_recording.mp4",
    )
