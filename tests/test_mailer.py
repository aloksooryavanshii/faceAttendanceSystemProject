import ssl

from face_attendance.config import Settings
from face_attendance.mailer import Mailer


class FakeSMTP:
    instances = []

    def __init__(self, host, port, context=None, timeout=None):
        self.args = (host, port, context, timeout)
        self.sent, self.login_args = [], None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, pw):
        self.login_args = (user, pw)

    def send_message(self, msg):
        self.sent.append(msg)


def cfg(tmp_path, **kw):
    base = dict(email_enabled=True, email_sender="a@x.com", email_password="pw", email_receiver="t@x.com")
    base.update(kw)
    return Settings.for_directory(tmp_path, **base)


def test_disabled_does_nothing(tmp_path):
    FakeSMTP.instances.clear()
    assert Mailer(cfg(tmp_path, email_enabled=False), FakeSMTP).send("s", "b") is False
    assert FakeSMTP.instances == []


def test_send_uses_verified_tls_and_timeout(tmp_path):
    FakeSMTP.instances.clear()
    assert Mailer(cfg(tmp_path), FakeSMTP).send("Attendance: ALOK", "body") is True
    host, port, context, timeout = FakeSMTP.instances[0].args
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname   # old code: CERT_NONE
    assert timeout == 15.0 and (host, port) == ("smtp.gmail.com", 465)
    msg = FakeSMTP.instances[0].sent[0]
    assert msg["To"] == "t@x.com" and msg["Subject"] == "Attendance: ALOK"


def test_missing_credentials_and_errors_never_raise(tmp_path):
    assert Mailer(cfg(tmp_path, email_password=""), FakeSMTP).send("s", "b") is False

    class Boom(FakeSMTP):
        def login(self, *a):
            raise OSError("network down")
    assert Mailer(cfg(tmp_path), Boom).send("s", "b") is False


def test_header_injection_is_rejected(tmp_path):
    FakeSMTP.instances.clear()
    assert Mailer(cfg(tmp_path), FakeSMTP).send("hi\nBcc: evil@x.com", "b") is False
    assert all(not i.sent for i in FakeSMTP.instances)


def test_send_async_returns_thread(tmp_path):
    FakeSMTP.instances.clear()
    t = Mailer(cfg(tmp_path), FakeSMTP).send_async("s", "b")
    t.join(3)
    assert FakeSMTP.instances[0].sent
    assert Mailer(cfg(tmp_path, email_enabled=False), FakeSMTP).send_async("s", "b") is None
