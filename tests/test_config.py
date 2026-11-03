import pytest

from face_attendance.config import ConfigError, load_settings, parse_env_file


def test_defaults_are_anchored_to_base_dir_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir("/")
    s = load_settings(tmp_path, environ={})
    assert s.images_dir == tmp_path / "images"
    assert s.attendance_csv == tmp_path / "attendance.csv"
    assert s.face_threshold == 0.50 and s.web_debug is False and s.web_host == "127.0.0.1"
    assert s.email_enabled is False and s.email_ready is False


def test_env_file_and_environment_precedence(tmp_path):
    (tmp_path / ".env").write_text("# comment\nFACE_THRESHOLD=0.45\nEMAIL_SENDER='a@b.c'\nexport WEB_PORT=6000\n")
    s = load_settings(tmp_path, environ={"FACE_THRESHOLD": "0.40"})
    assert s.face_threshold == 0.40            # real environment wins over .env
    assert s.email_sender == "a@b.c" and s.web_port == 6000


def test_password_is_not_in_repr(tmp_path):
    s = load_settings(tmp_path, environ={"EMAIL_PASSWORD": "super-secret"})
    assert "super-secret" not in repr(s)


@pytest.mark.parametrize("env", [{"FACE_THRESHOLD": "abc"}, {"FACE_THRESHOLD": "5"},
                                 {"EMAIL_ENABLED": "maybe"}, {"WEB_PORT": "0"}])
def test_invalid_values_raise_clear_error(tmp_path, env):
    with pytest.raises(ConfigError):
        load_settings(tmp_path, environ=env)


def test_email_ready_requires_all_fields(tmp_path):
    env = {"EMAIL_ENABLED": "true", "EMAIL_SENDER": "a@b.c", "EMAIL_RECEIVER": "t@b.c"}
    assert load_settings(tmp_path, environ=env).email_ready is False
    assert load_settings(tmp_path, environ={**env, "EMAIL_PASSWORD": "x"}).email_ready is True


def test_parse_env_file_missing(tmp_path):
    assert parse_env_file(tmp_path / "nope.env") == {}
