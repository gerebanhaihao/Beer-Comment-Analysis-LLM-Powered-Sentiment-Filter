from beer_sentiment.credentials import load_saved_keys, save_key


def test_saved_keys_round_trip_encrypted_for_current_windows_user(tmp_path):
    path = tmp_path / "credentials.json"
    save_key("deepseek", "secret-deepseek", path)
    save_key("qwen", "secret-qwen", path)

    contents = path.read_text(encoding="utf-8")
    assert "secret-deepseek" not in contents
    assert "secret-qwen" not in contents
    assert load_saved_keys(path) == {
        "deepseek": "secret-deepseek",
        "qwen": "secret-qwen",
    }
