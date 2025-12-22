import json
import logging
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings


logger = logging.getLogger(__name__)


class S3Manager:
    # Работаем с s3
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=boto3.session.Config(signature_version="s3v4"),
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"Bucket {self.bucket} already exists")
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info(f"Bucket {self.bucket} created")

    def save_model(self, model_id, model_data):
        try:
            key = f"models/{model_id}.joblib"
            self.client.put_object(Bucket=self.bucket, Key=key, Body=model_data)
            logger.info(f"Model {model_id} saved to S3")
            return True
        except Exception as e:
            logger.error(f"Failed to save model to S3: {e}")
            return False

    def load_model(self, model_id):
        try:
            key = f"models/{model_id}.joblib"
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Model {model_id} not found in S3")
            else:
                logger.error(f"Error loading model from S3: {e}")
            return None

    def delete_model(self, model_id):
        try:
            key = f"models/{model_id}.joblib"
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Model {model_id} deleted from S3")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model from S3: {e}")
            return False

    def save_metadata(self, metadata):
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key="models_metadata.json",
                Body=json.dumps(metadata).encode("utf-8"),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to S3: {e}")
            return False

    def load_metadata(self):
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key="models_metadata.json"
            )
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError:
            return {}
