#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path


PROGRAM_PATH = Path(__file__).resolve().parent / "program.md"


def extract_credential(label: str, text: str) -> str:
    pattern = rf"- {re.escape(label)}: `([^`]+)`"
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Missing credential field in program.md: {label}")
    return match.group(1)


def load_gmail_config() -> tuple[str, str, str]:
    text = PROGRAM_PATH.read_text(encoding="utf-8")
    sender = extract_credential("Gmail SMTP sender", text)
    recipient = extract_credential("Gmail SMTP recipient", text)
    password = extract_credential("Gmail app password", text)
    return sender, recipient, password


def _split_body_lines(body: str) -> list[str]:
    return [line.rstrip() for line in body.strip().splitlines()]


def _field_rows(lines: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in lines:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if not key.strip() or not value.strip():
            continue
        rows.append((key.strip(), value.strip()))
    return rows


def _bullet_items(lines: list[str]) -> list[str]:
    return [line[2:].strip() for line in lines if line.startswith("- ") and line[2:].strip()]


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("- "):
            continue
        if ": " in line:
            continue
        paragraphs.append(line.strip())
    return paragraphs


def build_html_email(subject: str, body: str) -> str:
    lines = _split_body_lines(body)
    fields = _field_rows(lines)
    bullets = _bullet_items(lines)
    paragraphs = _paragraphs(lines)

    def metric_row(label: str, value: str) -> str:
        return (
            "<tr>"
            f"<td class='card label' bgcolor='#13202f' style='padding:10px 12px;color:#9fb2c8 !important;font-size:13px;font-weight:600;"
            "border-bottom:1px solid #2a3b51;vertical-align:top;background:#13202f !important;'>"
            f"{html.escape(label)}</td>"
            f"<td class='card' bgcolor='#13202f' style='padding:10px 12px;color:#edf4fb !important;font-size:14px;line-height:1.45;"
            "border-bottom:1px solid #2a3b51;background:#13202f !important;'>"
            f"{html.escape(value)}</td>"
            "</tr>"
        )

    rows_html = "".join(metric_row(label, value) for label, value in fields)
    bullets_html = "".join(
        (
            "<li style='margin:0 0 8px 0;color:#edf4fb !important;font-size:14px;line-height:1.5;'>"
            f"{html.escape(item)}</li>"
        )
        for item in bullets
    )
    paragraphs_html = "".join(
        (
            "<p style='margin:0 0 12px 0;color:#edf4fb !important;font-size:14px;line-height:1.6;'>"
            f"{html.escape(item)}</p>"
        )
        for item in paragraphs
    )
    bullets_list_html = (
        "<ul style='padding-left:20px;margin:6px 0 0 0;'>"
        f"{bullets_html}"
        "</ul>"
        if bullets_html
        else ""
    )

    summary_block = ""
    if rows_html:
        summary_block = (
            "<div class='card' bgcolor='#13202f' style='background:#13202f !important;border:1px solid #2a3b51;border-radius:16px;"
            "overflow:hidden;box-shadow:0 8px 24px rgba(18,34,52,0.08);'>"
            "<div class='card-head' bgcolor='#1a2d42' style='padding:14px 16px;background:#1a2d42 !important;color:#d9e5f3 !important;font-size:12px;"
            "font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'>Run Summary</div>"
            "<table class='card' role='presentation' width='100%' cellspacing='0' cellpadding='0' bgcolor='#13202f' style='border-collapse:collapse;background:#13202f !important;'>"
            f"{rows_html}"
            "</table>"
            "</div>"
        )

    notes_block = ""
    if paragraphs_html or bullets_html:
        notes_block = (
            "<div class='card' bgcolor='#13202f' style='margin-top:18px;background:#13202f !important;border:1px solid #2a3b51;border-radius:16px;"
            "padding:18px 20px;box-shadow:0 8px 24px rgba(18,34,52,0.08);'>"
            "<div class='card-head' bgcolor='#13202f' style='margin:0 0 12px 0;color:#d9e5f3 !important;font-size:12px;font-weight:700;"
            "letter-spacing:0.08em;text-transform:uppercase;background:#13202f !important;'>Agent Notes</div>"
            f"{paragraphs_html}"
            f"{bullets_list_html}"
            "</div>"
        )

    return f"""\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      body, table, td, div, p, li, span {{
        color: #edf4fb !important;
      }}
      .bg-body {{
        background: #0b1220 !important;
      }}
      .hero {{
        background: #12243c !important;
      }}
      .card {{
        background: #13202f !important;
      }}
      .card-head {{
        background: #1a2d42 !important;
        color: #d9e5f3 !important;
      }}
      .label {{
        color: #9fb2c8 !important;
      }}
      .muted {{
        color: #c6d3e3 !important;
      }}
      .footer {{
        color: #b0c0d3 !important;
      }}
    </style>
  </head>
  <body class="bg-body" bgcolor="#0b1220" style="margin:0;padding:0;background:#0b1220 !important;color:#edf4fb !important;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    <div class="bg-body" style="padding:28px 12px;background:#0b1220 !important;">
      <div style="max-width:720px;margin:0 auto;">
        <div class="hero" bgcolor="#12243c" style="background:#12243c !important;border-radius:22px;
          padding:24px 24px 20px 24px;box-shadow:0 16px 40px rgba(13,27,42,0.22);">
          <div bgcolor="#223a5c" style="display:inline-block;padding:6px 10px;border-radius:999px;background:#223a5c !important;
            color:#e8f0fa !important;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">
            Research Agent
          </div>
          <div style="margin-top:14px;color:#ffffff !important;font-size:28px;font-weight:800;letter-spacing:-0.02em;">
            Parameter Golf Autoresearch Update
          </div>
          <div class="muted" style="margin-top:8px;color:#c6d3e3 !important;font-size:14px;line-height:1.5;">
            {html.escape(subject)}
          </div>
        </div>

        <div style="margin-top:18px;">
          {summary_block}
          {notes_block}
        </div>

        <div class="footer" style="margin-top:14px;padding:0 4px;color:#b0c0d3 !important;font-size:12px;line-height:1.5;">
          Auto-generated by the local research loop on Seeweb.
        </div>
      </div>
    </div>
  </body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Send autoresearch update email through Gmail SMTP.")
    parser.add_argument("--subject", required=True, help="Email subject")
    parser.add_argument("--body-file", help="Path to a UTF-8 text file containing the email body")
    parser.add_argument("--body", help="Inline email body")
    args = parser.parse_args()

    if bool(args.body_file) == bool(args.body):
        raise SystemExit("Provide exactly one of --body-file or --body")

    body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else args.body
    sender, recipient, password = load_gmail_config()
    subject = args.subject if args.subject.lower().startswith("research agent") else f"Research Agent | {args.subject}"

    msg = EmailMessage()
    msg["From"] = formataddr(("Research Agent", sender))
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_alternative(build_html_email(subject, body), subtype="html")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)


if __name__ == "__main__":
    main()
