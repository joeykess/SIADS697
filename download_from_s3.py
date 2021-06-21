import boto3


def download_files():
    """
    Downloads all s3 files in bucket to local
    """
    bucket = "assets-bucket-q2wc9h6fcov44k5"
    next_token = ''
    s3 = boto3.client('s3')
    while next_token is not None:
        if next_token != '':
            cont_token = next_token
        results = s3.list_objects_v2(Bucket=bucket)
        print(results)


if __name__ == '__main__':
    download_files()
