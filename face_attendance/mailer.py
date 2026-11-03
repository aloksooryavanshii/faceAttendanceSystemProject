"""E-mail notification (Gmail SMTP over SSL, with certificate verification and a timeout)."""
from __future__ import annotations

import logging
import smtplib
import ssl
import threading
from email.message import EmailMessage
from typing import Callable, Optional

from .config import Settings

log = logging.getLogger(__name__)


class Mailer:
    def __init__(self, settings: Settings, smtp_factory: Callable[..., smtplib.SMTP_SSL] = smtplib.SMTP_SSL):
        self._s = settings
        self._smtp_factory = smtp_factory

    def send(self, subject: str, body: str) -> bool:
        """Send one plain-text mail. Returns True on success; never raises."""
        s = self._s
        if not s.email_enabled:
            return False
        if not s.email_ready:
            log.error("E-mail is enabled but EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECEIVER are not all set.")
            return False
        try:
            msg = EmailMessage()                 # rejects header injection (newlines) with ValueError
            msg["From"], msg["To"], msg["Subject"] = s.email_sender, s.email_receiver, subject
            msg.set_content(body)
            context = ssl.create_default_context()   # verifies the server certificate
            with self._smtp_factory(s.smtp_host, s.smtp_port, context=context,
                                    timeout=s.smtp_timeout) as server:
                server.login(s.email_sender, s.email_password)
                server.send_message(msg)
            log.info("[EMAIL] Sent: %s", subject)
            return True
        except Exception as exc:  # network, auth, bad header ... must not kill the camera loop
            log.error("[EMAIL ERROR] %s", exc)
            return False

    def send_async(self, subject: str, body: str) -> Optional[threading.Thread]:
        if not self._s.email_enabled:
            return None
        thread = threading.Thread(target=self.send, args=(subject, body), daemon=True, name="mailer")
        thread.start()
        return thread
