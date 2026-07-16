from hashlib import sha256
from pathlib import Path

from face_classifier_service.analysis import UnavailableAnalyzer, create_default_analyzer
from face_classifier_service.models import MODEL_MANIFEST
from face_classifier_service.settings import Settings


def write_manifest_files(models_dir: Path) -> None:
    for artifact in MODEL_MANIFEST:
        path = models_dir / artifact.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"model"
        path.write_bytes(payload)
        path.with_name(f"{path.name}.sha256").write_text(
            sha256(payload).hexdigest(),
            encoding="ascii",
        )


def test_default_analyzer_uses_configured_factory_after_model_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_manifest_files(tmp_path)
    module_path = tmp_path / "custom_factory.py"
    module_path.write_text(
        "class ReadyAnalyzer:\n"
        "    ready = True\n"
        "    def __init__(self, reason):\n"
        "        self.reason = reason\n"
        "    def analyze(self, image, options):\n"
        "        raise RuntimeError(self.reason)\n"
        "def build(settings):\n"
        "    return ReadyAnalyzer(f'factory:{settings.models_dir}')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    analyzer = create_default_analyzer(
        Settings(models_dir=tmp_path, analyzer_factory="custom_factory:build")
    )

    assert analyzer.reason == f"factory:{tmp_path}"


def test_default_analyzer_reports_missing_models_before_loading_factory(tmp_path: Path) -> None:
    analyzer = create_default_analyzer(
        Settings(models_dir=tmp_path, analyzer_factory="missing_module:build")
    )

    assert isinstance(analyzer, UnavailableAnalyzer)
    assert "invalid local model artifacts" in analyzer.reason


def test_default_analyzer_reports_runtime_factory_failure(tmp_path: Path, monkeypatch) -> None:
    write_manifest_files(tmp_path)
    module_path = tmp_path / "failing_factory.py"
    module_path.write_text(
        "def build(settings):\n"
        "    raise RuntimeError('cuda provider failed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    analyzer = create_default_analyzer(
        Settings(models_dir=tmp_path, analyzer_factory="failing_factory:build")
    )

    assert isinstance(analyzer, UnavailableAnalyzer)
    assert "failed to load analyzer factory: cuda provider failed" == analyzer.reason
