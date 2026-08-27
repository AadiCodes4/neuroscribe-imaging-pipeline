"""
S3-compatible storage helper built on boto3.

This is written to work against any S3-compatible endpoint (real AWS S3,
MinIO, LocalStack, etc.) by pointing `endpoint_url` at it, and is verified in
this project's test suite using `moto`'s in-memory mock AWS backend — NOT
against real AWS. No real AWS credentials or bucket were used or are
required to develop or test this module.

Typical uses in this project: persisting uploaded scan images and their
resulting segmentation masks / saliency maps as objects, keyed by a
request/session id.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


@dataclass
class StorageConfig:
    bucket_name: str = "neuroscribe-artifacts"
    region_name: str = "us-east-1"
    endpoint_url: Optional[str] = None  # e.g. "http://localhost:9000" for MinIO
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "StorageConfig":
        return cls(
            bucket_name=os.environ.get("NEUROSCRIBE_S3_BUCKET", "neuroscribe-artifacts"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            endpoint_url=os.environ.get("NEUROSCRIBE_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )


class StorageClient:
    """Thin, testable wrapper around a boto3 S3 client."""

    def __init__(self, config: Optional[StorageConfig] = None, client: Optional[BaseClient] = None):
        self.config = config or StorageConfig.from_env()
        self._client = client or boto3.client(
            "s3",
            region_name=self.config.region_name,
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

    def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self.config.bucket_name)
        except ClientError:
            create_kwargs = {"Bucket": self.config.bucket_name}
            if self.config.region_name and self.config.region_name != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.config.region_name
                }
            self._client.create_bucket(**create_kwargs)

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes under `key`. Returns the key on success."""
        self.ensure_bucket()
        self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def download_bytes(self, key: str) -> bytes:
        """Download and return the raw bytes stored under `key`."""
        obj = self._client.get_object(Bucket=self.config.bucket_name, Key=key)
        return obj["Body"].read()

    def object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.config.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.config.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete_key(self, key: str) -> None:
        self._client.delete_object(Bucket=self.config.bucket_name, Key=key)
