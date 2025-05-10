import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

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

def generate_s3_presigned_download_url(bucket_name, object_key, expiration=3600, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
    """
    Generates a pre-signed URL to download an S3 object.

    :param bucket_name: String name of the S3 bucket.
    :param object_key: String key of the S3 object (path to file).
    :param expiration: Integer time in seconds for the pre-signed URL to remain valid.
    :param region_name: String AWS region. If None, boto3 will try to determine the region.
    :param aws_access_key_id: String AWS access key ID. Optional.
    :param aws_secret_access_key: String AWS secret access key. Optional.
    :return: String pre-signed URL if successful, else None.
    """
    s3_client_args = {}
    if region_name:
        s3_client_args['region_name'] = region_name
    if aws_access_key_id and aws_secret_access_key:
        s3_client_args['aws_access_key_id'] = aws_access_key_id
        s3_client_args['aws_secret_access_key'] = aws_secret_access_key

    s3_client = boto3.client('s3', **s3_client_args)

    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return response
    except NoCredentialsError:
        print("Error: AWS credentials not found. Configure credentials or pass them to the function.")
        return None
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials found.")
        return None
    except ClientError as e:
        print(f"ClientError generating pre-signed URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Usage
    fetch_from_s3(
        bucket="talentorarecordings",
        key="af4be366-5f9a-46a1-a48c-0788363168c0/6590b339-db89-415a-bc62-b3599af6bc48/interview_recording.mp4",
        local_path="./interview_recording.mp4",
    )
