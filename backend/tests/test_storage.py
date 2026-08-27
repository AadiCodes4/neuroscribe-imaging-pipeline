"""
Tests for StorageClient using `moto`'s in-memory mock AWS backend.

These tests DO NOT touch real AWS. moto intercepts boto3 calls at the
botocore layer, so `StorageClient` is exercised exactly as it would be
against real S3 (or a real S3-compatible service), just backed by an
in-memory fake. No AWS credentials or network access are required.
"""

import boto3
import pytest
from moto import mock_aws

from app.storage import StorageClient, StorageConfig


@pytest.fixture()
def storage():
    with mock_aws():
        config = StorageConfig(bucket_name="neuroscribe-test-bucket", region_name="us-east-1")
        client = boto3.client("s3", region_name="us-east-1")
        yield StorageClient(config=config, client=client)


def test_ensure_bucket_is_idempotent(storage):
    storage.ensure_bucket()
    storage.ensure_bucket()  # should not raise even though it already exists
    assert storage.list_keys() == []


def test_upload_and_download_roundtrip(storage):
    key = "scans/sample-001/original.png"
    data = b"\x89PNG\r\n\x1a\nfake-png-bytes-for-testing"

    returned_key = storage.upload_bytes(key, data, content_type="image/png")
    assert returned_key == key

    downloaded = storage.download_bytes(key)
    assert downloaded == data


def test_object_exists(storage):
    key = "scans/sample-002/mask.png"
    assert storage.object_exists(key) is False

    storage.upload_bytes(key, b"mask-bytes")
    assert storage.object_exists(key) is True


def test_list_keys_with_prefix(storage):
    storage.upload_bytes("scans/a/original.png", b"a")
    storage.upload_bytes("scans/a/mask.png", b"a-mask")
    storage.upload_bytes("scans/b/original.png", b"b")

    a_keys = sorted(storage.list_keys(prefix="scans/a/"))
    assert a_keys == ["scans/a/mask.png", "scans/a/original.png"]

    all_keys = sorted(storage.list_keys())
    assert len(all_keys) == 3


def test_delete_key(storage):
    key = "scans/c/original.png"
    storage.upload_bytes(key, b"c")
    assert storage.object_exists(key) is True

    storage.delete_key(key)
    assert storage.object_exists(key) is False
