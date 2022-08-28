import boto3
from botocore.exceptions import NoCredentialsError

SECRET_KEY = 'ULGn7r0Yy/SvHBK0TPPOXyjSOALZv7T1gP3vY3Ir'
ACCESS_KEY = 'AKIA547RNSAVXOJWAYHC'
BUCKET = "helmet-detection-data"


def upload_to_aws(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, BUCKET, s3_file)
        print("Video enviado a s3::%s" % s3_file)
        return True
    except FileNotFoundError:
        print("The file was not found s3::%s" % s3_file)
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

# def upload_to_one_drive(local_file, s3_file):
