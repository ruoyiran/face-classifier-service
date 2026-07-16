from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nacl.exceptions import CryptoError
from nacl.public import PrivateKey, PublicKey, SealedBox

ENCRYPTION_ALGORITHM = "AES-256-GCM+SealedBox-v1"
AES_256_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12


@dataclass(frozen=True)
class EncryptionKeyPair:
    private_key_b64: str
    public_key_b64: str


@dataclass(frozen=True)
class EncryptedImageForServer:
    payload: dict[str, str]
    aes_key: bytes


@dataclass(frozen=True)
class DecryptedImagePayload:
    image_bytes: bytes
    aes_key: bytes


def generate_key_pair() -> EncryptionKeyPair:
    private_key = PrivateKey.generate()
    public_key = private_key.public_key
    return EncryptionKeyPair(
        private_key_b64=encode_b64(bytes(private_key)),
        public_key_b64=encode_b64(bytes(public_key)),
    )


def write_key_pair(private_key_path: Path, public_key_path: Path) -> EncryptionKeyPair:
    key_pair = generate_key_pair()
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_text(key_pair.private_key_b64, encoding="ascii")
    public_key_path.write_text(key_pair.public_key_b64, encoding="ascii")
    return key_pair


def read_key_b64(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"encryption key file does not exist: {path}")
    return path.read_text(encoding="ascii").strip()


def encrypt_image_for_server(public_key_b64: str, image_bytes: bytes) -> dict[str, str]:
    return encrypt_image_for_server_with_key(public_key_b64, image_bytes).payload


def encrypt_image_for_server_with_key(
    public_key_b64: str,
    image_bytes: bytes,
) -> EncryptedImageForServer:
    aes_key = os.urandom(AES_256_KEY_BYTES)
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    ciphertext = AESGCM(aes_key).encrypt(nonce, image_bytes, None)
    encrypted_key = SealedBox(PublicKey(decode_b64(public_key_b64))).encrypt(aes_key)
    return EncryptedImageForServer(
        payload={
            "alg": ENCRYPTION_ALGORITHM,
            "encrypted_key": encode_b64(encrypted_key),
            "nonce": encode_b64(nonce),
            "ciphertext": encode_b64(ciphertext),
        },
        aes_key=aes_key,
    )


def decrypt_encrypted_image_payload(payload: dict[str, Any], private_key_b64: str) -> bytes:
    return decrypt_encrypted_image_payload_with_key(payload, private_key_b64).image_bytes


def decrypt_encrypted_image_payload_with_key(
    payload: dict[str, Any],
    private_key_b64: str,
) -> DecryptedImagePayload:
    if payload.get("alg") != ENCRYPTION_ALGORITHM:
        raise ValueError("unsupported encryption algorithm")

    try:
        encrypted_key = decode_b64(str(payload["encrypted_key"]))
        nonce = decode_b64(str(payload["nonce"]))
        ciphertext = decode_b64(str(payload["ciphertext"]))
        private_key = PrivateKey(decode_b64(private_key_b64))
        aes_key = SealedBox(private_key).decrypt(encrypted_key)
        if len(aes_key) != AES_256_KEY_BYTES or len(nonce) != AES_GCM_NONCE_BYTES:
            raise ValueError("invalid encrypted payload")
        image_bytes = AESGCM(aes_key).decrypt(nonce, ciphertext, None)
        return DecryptedImagePayload(image_bytes=image_bytes, aes_key=aes_key)
    except (KeyError, binascii.Error, CryptoError, InvalidTag, ValueError) as exc:
        if str(exc) == "unsupported encryption algorithm":
            raise
        raise ValueError("invalid encrypted payload") from exc


def encrypt_response_for_client(
    *,
    aes_key: bytes,
    response_payload: dict[str, Any],
    key_index: int,
) -> dict[str, Any]:
    if len(aes_key) != AES_256_KEY_BYTES:
        raise ValueError("invalid response encryption key")
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    plaintext = json.dumps(
        response_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, None)
    return {
        "alg": ENCRYPTION_ALGORITHM,
        "key_index": key_index,
        "nonce": encode_b64(nonce),
        "ciphertext": encode_b64(ciphertext),
    }


def decrypt_response_from_server(payload: dict[str, Any], aes_key: bytes) -> dict[str, Any]:
    if payload.get("alg") != ENCRYPTION_ALGORITHM:
        raise ValueError("unsupported encryption algorithm")
    try:
        nonce = decode_b64(str(payload["nonce"]))
        ciphertext = decode_b64(str(payload["ciphertext"]))
        if len(aes_key) != AES_256_KEY_BYTES or len(nonce) != AES_GCM_NONCE_BYTES:
            raise ValueError("invalid encrypted response")
        plaintext = AESGCM(aes_key).decrypt(nonce, ciphertext, None)
        decoded = json.loads(plaintext.decode("utf-8"))
    except (
        KeyError,
        binascii.Error,
        InvalidTag,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        if str(exc) == "unsupported encryption algorithm":
            raise
        raise ValueError("invalid encrypted response") from exc
    if not isinstance(decoded, dict):
        raise ValueError("invalid encrypted response")
    return decoded


def encode_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_b64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)
