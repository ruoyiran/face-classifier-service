from face_classifier_service import auth


def test_token_matches_checks_all_configured_tokens(monkeypatch) -> None:
    compared: list[tuple[str, str]] = []

    def fake_compare_digest(left: str, right: str) -> bool:
        compared.append((left, right))
        return right == "primary-token"

    monkeypatch.setattr(auth, "compare_digest", fake_compare_digest)

    assert auth.token_matches("provided-token", ("primary-token", "secondary-token")) is True
    assert compared == [
        ("provided-token", "primary-token"),
        ("provided-token", "secondary-token"),
    ]
