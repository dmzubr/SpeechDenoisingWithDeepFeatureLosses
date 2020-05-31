import boto3


class YaCloudService:
    YA_ENDPOINT_URL = 'https://storage.yandexcloud.net'

    def __init__(self, bucket_name, aws_creds_obj):
        self.__bucket_name = bucket_name
        self.__aws_creds_obj = aws_creds_obj

    def save_object_to_storage(self, file_to_save, save_file_name):
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url=self.YA_ENDPOINT_URL,
            aws_access_key_id=self.__aws_creds_obj['aws_access_key_id'],
            aws_secret_access_key=self.__aws_creds_obj['aws_secret_access_key']
        )

        s3.upload_file(file_to_save, self.__bucket_name, save_file_name)

        res = f'{self.YA_ENDPOINT_URL}/{self.__bucket_name}/{save_file_name}'
        return res

