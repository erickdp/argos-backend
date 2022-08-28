import os
import subprocess

import boto3
from botocore.exceptions import NoCredentialsError

SECRET_KEY = 'ULGn7r0Yy/SvHBK0TPPOXyjSOALZv7T1gP3vY3Ir' # saul
ACCESS_KEY = 'AKIA547RNSAVXOJWAYHC'
BUCKET = "helmet-detection-data"

# SECRET_KEY = 'tPV7sNZ0x8DRpX8eH/xzHnKkkrCgyDGnUquK9Hu9' # erick
# ACCESS_KEY = 'AKIA4DDPPWB4TC332FTC'
# BUCKET = "ucemineriabucket"

from stream_service import source_file


def upload_to_aws(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    subprocess.run(['ffmpeg', '-i', f'{source_file}/{local_file}', f'{source_file}/f{local_file}'])
    try:
        s3.upload_file(f'{source_file}/{local_file}', BUCKET, s3_file, ExtraArgs={'ContentType': "video/mp4"})
        os.remove(f'{source_file}/{local_file}')
        os.remove(f'{source_file}/f{local_file}')
        print("Video enviado a s3::%s" % s3_file)
        return True
    except FileNotFoundError:
        print("The file was not found s3::%s" % s3_file)
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False



# def upload_to_one_drive(local_file, s3_file):

# if __name__ == '__main__':
#     subprocess.run(['ffmpeg', '-i', './videos/as2844111.mp4', './videos/fas2844111.mp4'])
#     upload_to_aws("./videos/fas2844111.mp4", 'ffas2844111.mp4')
#     os.remove(["dsf", "fsda"])
