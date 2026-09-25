"""
storage/r2_client.py — thin wrapper around Cloudflare R2 (S3-compatible).
Falls back to local disk if R2 credentials aren't set.
"""
from __future__ import annotations
from typing import Optional

from config import settings, Paths


def _using_r2() -> bool:
    return bool(settings.r2_access_key and settings.r2_secret_key
                and settings.r2_bucket and settings.r2_endpoint)


def _get_r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key,
        aws_secret_access_key=settings.r2_secret_key,
        region_name="auto",
    )


def upload_bytes(key: str, data: bytes) -> bool:
    if not _using_r2():
        path = Paths.MODELS / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True
    try:
        _get_r2_client().put_object(Bucket=settings.r2_bucket, Key=key, Body=data)
        return True
    except Exception as e:
        print(f"[r2] upload('{key}') failed: {e}")
        return False


def download_bytes(key: str) -> Optional[bytes]:
    if not _using_r2():
        path = Paths.MODELS / key
        return path.read_bytes() if path.exists() else None
    try:
        obj = _get_r2_client().get_object(Bucket=settings.r2_bucket, Key=key)
        return obj["Body"].read()
    except Exception as e:
        print(f"[r2] download('{key}') failed (may not exist yet): {e}")
        return None