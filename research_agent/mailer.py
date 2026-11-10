"""MIME construction, address validation and SMTP transport with TLS."""
from __future__ import annotations

from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import os
import re
import smtplib
import ssl

from .config import env_int
from .digest import render_digest
from .models import Evaluated


def create_email(items: list[Evaluated], date: str, scanned: int, threshold: int,
                 sender: str, recipients: list[str]) -> EmailMessage:
    """Build a MIME message with plain-text and HTML versions without sending it."""
    plain, rich = render_digest(items, date, scanned, threshold)
    msg = EmailMessage()
    msg["Subject"] = f"[Research radar] {len(items)} relevant arXiv paper(s) - {date}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=None)
    # Clients without HTML support receive the same information as plain text.
    msg.set_content(plain)
    msg.add_alternative(rich, subtype="html")
    return msg


def send_email(msg: EmailMessage) -> None:
    """Send over SMTP with implicit TLS on port 465 and STARTTLS on other ports.

    Connection, authentication and delivery errors propagate to prevent the
    pipeline from advancing reviewed history when relay delivery fails.
    """
    host = os.getenv("SMTP_HOST", "").strip()
    port = env_int("SMTP_PORT", 587, 1, 65535)
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    if not host:
        raise ValueError("Set SMTP_HOST in .env before sending")
    if user and not password:
        raise ValueError("SMTP_USER is set but SMTP_PASSWORD is empty")
    tls = ssl.create_default_context()
    if port == 465:
        client = smtplib.SMTP_SSL(host, port, timeout=30, context=tls)
    else:
        client = smtplib.SMTP(host, port, timeout=30)
    with client as smtp:
        # Authenticate only after establishing the encrypted connection.
        if port != 465:
            smtp.ehlo()
            smtp.starttls(context=tls)
            smtp.ehlo()
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)


def valid_email(value: str) -> bool:
    """Check the basic format of an email address before preparing delivery."""
    return bool(re.fullmatch(r"[^\s@,<>]+@[^\s@,<>]+\.[^\s@,<>]+", value))
