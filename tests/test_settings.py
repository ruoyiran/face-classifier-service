from pathlib import Path

import pytest
from pydantic import ValidationError

from face_classifier_service.settings import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("/", ""),
        ("/face-api", "/face-api"),
        ("/face-api/", "/face-api"),
    ],
)
def test_api_base_path_is_normalized(raw: str, expected: str) -> None:
    settings = Settings(api_base_path=raw)

    assert settings.api_base_path == expected


@pytest.mark.parametrize("raw", ["face api", "/face-api?x=1", "/face-api#fragment"])
def test_api_base_path_rejects_invalid_url_path_prefixes(raw: str) -> None:
    with pytest.raises(ValidationError):
        Settings(api_base_path=raw)


def test_api_base_path_reads_api_base_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_PATH", "/face-api")

    settings = Settings()

    assert settings.api_base_path == "/face-api"


def test_model_directory_defaults_to_absolute_models_path() -> None:
    settings = Settings()

    assert settings.models_dir == Path("/models")


def test_batch_and_image_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)

    with pytest.raises(ValidationError):
        Settings(max_image_bytes=0)


def test_api_tokens_are_parsed_from_comma_separated_string() -> None:
    settings = Settings(api_tokens="primary-token, secondary-token,")

    assert settings.api_tokens == ("primary-token", "secondary-token")


def test_auth_requires_at_least_one_token_when_enabled() -> None:
    with pytest.raises(ValidationError):
        Settings(auth_enabled=True)


def test_encrypted_limits_and_key_paths_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.max_encrypted_image_bytes == 42 * 1024 * 1024
    assert settings.encryption_private_key_path.as_posix() == "config/server_private.key"
    assert settings.encryption_public_key_path.as_posix() == "config/server_public.key"
