import pytest

from face_classifier_service.encryption import (
    ENCRYPTION_ALGORITHM,
    decrypt_encrypted_image_payload,
    decrypt_response_from_server,
    encrypt_image_for_server,
    encrypt_image_for_server_with_key,
    encrypt_response_for_client,
    generate_key_pair,
)


def test_hybrid_encryption_round_trip() -> None:
    key_pair = generate_key_pair()

    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )

    assert payload["alg"] == ENCRYPTION_ALGORITHM
    assert decrypt_encrypted_image_payload(payload, key_pair.private_key_b64) == b"fake-image"


def test_hybrid_encryption_can_return_client_aes_key_for_response_decryption() -> None:
    key_pair = generate_key_pair()

    encrypted = encrypt_image_for_server_with_key(key_pair.public_key_b64, b"fake-image")

    assert encrypted.payload["alg"] == ENCRYPTION_ALGORITHM
    assert encrypted.aes_key
    assert (
        decrypt_encrypted_image_payload(encrypted.payload, key_pair.private_key_b64)
        == b"fake-image"
    )


def test_hybrid_encryption_rejects_tampered_ciphertext() -> None:
    key_pair = generate_key_pair()
    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )
    payload["ciphertext"] = payload["ciphertext"][:-2] + "AA"

    with pytest.raises(ValueError, match="invalid encrypted payload"):
        decrypt_encrypted_image_payload(payload, key_pair.private_key_b64)


def test_hybrid_encryption_rejects_unknown_algorithm() -> None:
    key_pair = generate_key_pair()
    payload = encrypt_image_for_server(
        public_key_b64=key_pair.public_key_b64,
        image_bytes=b"fake-image",
    )
    payload["alg"] = "AES-256-CBC"

    with pytest.raises(ValueError, match="unsupported encryption algorithm"):
        decrypt_encrypted_image_payload(payload, key_pair.private_key_b64)


def test_response_encryption_round_trips_json_with_existing_aes_key() -> None:
    key_pair = generate_key_pair()
    encrypted = encrypt_image_for_server_with_key(key_pair.public_key_b64, b"fake-image")
    response_payload = {"results": [{"image_id": "one.jpg", "faces": []}]}

    encrypted_response = encrypt_response_for_client(
        aes_key=encrypted.aes_key,
        response_payload=response_payload,
        key_index=0,
    )

    assert encrypted_response["alg"] == ENCRYPTION_ALGORITHM
    assert encrypted_response["key_index"] == 0
    assert "results" not in encrypted_response
    assert decrypt_response_from_server(encrypted_response, encrypted.aes_key) == response_payload
