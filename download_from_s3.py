import boto3, os


def download_files():
    """
    Downloads all s3 files in bucket to local
    """
    bucket = "assets-bucket-q2wc9h6fcov44k5"
    profile = input("Input your profile name")
    boto3.setup_default_session(profile_name=profile)
    s3 = boto3.client('s3')
    results = s3.list_objects_v2(Bucket=bucket)
    results = [item['Key'] for item in results['Contents']]
    path = os.path.join(os.getcwd(), 'assets')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not os.path.exists(os.path.dirname(path+'historical-symbols')):
        os.makedirs(os.path.join(path, 'historical-symbols'))
    for folder in results:
        dest_path = path + '/' + folder
        s3.download_file(bucket, folder, dest_path)


if __name__ == '__main__':
    download_files()
