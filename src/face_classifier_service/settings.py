from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FACE_",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "face-classifier-service"
    host: str = "127.0.0.1"
    port: int = 8000
    api_base_path: str = Field(
        default="",
        validation_alias=AliasChoices("API_BASE_PATH", "FACE_API_BASE_PATH"),
    )
    analyzer_factory: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANALYZER_FACTORY", "FACE_ANALYZER_FACTORY"),
    )
    models_dir: Path = Field(
        default=Path("/models"),
        validation_alias=AliasChoices("MODELS_DIR", "FACE_MODELS_DIR"),
    )
    device: str = "cuda"
    max_image_bytes: int = 30 * 1024 * 1024
    max_encrypted_image_bytes: int = 42 * 1024 * 1024
    max_multipart_overhead_bytes: int = 64 * 1024
    max_image_pixels: int = 16_000_000
    max_batch_size: int = 8
    allow_debug_embedding_json: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ALLOW_DEBUG_EMBEDDING_JSON",
            "FACE_ALLOW_DEBUG_EMBEDDING_JSON",
        ),
    )
    auth_enabled: bool = False
    api_tokens: Annotated[tuple[str, ...], NoDecode] = ()
    auth_disable_docs: bool = True
    encryption_enabled: bool = False
    require_encrypted_uploads: bool = False
    encryption_private_key_path: Path = Path("config/server_private.key")
    encryption_public_key_path: Path = Path("config/server_public.key")

    @field_validator("api_base_path")
    @classmethod
    def normalize_api_base_path(cls, value: str) -> str:
        value = value.strip()
        if value in {"", "/"}:
            return ""
        if " " in value or "?" in value or "#" in value:
            raise ValueError(
                "api_base_path must be a URL path prefix without spaces, query, or fragment"
            )
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    @field_validator(
        "max_image_bytes",
        "max_encrypted_image_bytes",
        "max_multipart_overhead_bytes",
        "max_image_pixels",
    )
    @classmethod
    def validate_positive_image_limits(cls, value: int, info):
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_batch_size must be >= 1")
        return value

    @field_validator("api_tokens", mode="before")
    @classmethod
    def parse_api_tokens(cls, value):
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(token.strip() for token in value.split(",") if token.strip())
        return tuple(str(token).strip() for token in value if str(token).strip())

    @model_validator(mode="after")
    def validate_auth_configuration(self):
        if self.auth_enabled and not self.api_tokens:
            raise ValueError("api_tokens must contain at least one token when auth_enabled is true")
        return self
