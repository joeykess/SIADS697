import boto3


def download_files():
    """
    Downloads all s3 files in bucket to local
    """
    bucket = "assets-bucket-q2wc9h6fcov44k5"
    s3 = boto3.client('s3')
    results = s3.list_objects_v2(Bucket=bucket)
    results = [item['Key'] for item in results['Contents']]
    print(results)


if __name__ == '__main__':
    download_files()
