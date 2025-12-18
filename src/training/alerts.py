"""
SMS/Email alerting system for training monitoring.

Supports two methods:
1. Twilio API (recommended) - reliable direct SMS
2. Email-to-SMS gateways (fallback) - free but unreliable

Supports quiet hours for non-critical alerts.
"""

import smtplib
from email.message import EmailMessage
from datetime import datetime, time
from typing import Optional
from enum import Enum
import os

# Try to import twilio
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # NaN, crash, explosion - always send
    CHECKPOINT = "checkpoint"  # Checkpoint saved - respects quiet hours
    PROGRESS = "progress"  # Hourly update - respects quiet hours
    COMPLETE = "complete"  # Training finished - always send


class TrainingAlerts:
    """
    SMS alerting via Twilio API or email-to-SMS gateway.

    Twilio (recommended):
        Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER env vars
        or pass twilio_* parameters

    Email-to-SMS gateways (fallback):
        - T-Mobile: number@tmomail.net
        - Verizon: number@vtext.com
        - AT&T: number@txt.att.net
        - Sprint: number@messaging.sprintpcs.com
    """

    CARRIER_GATEWAYS = {
        "tmobile": "tmomail.net",
        "verizon": "vtext.com",
        "att": "txt.att.net",
        "sprint": "messaging.sprintpcs.com",
    }

    def __init__(
        self,
        phone_number: str,
        carrier: str = "tmobile",
        smtp_host: str = "localhost",
        smtp_port: int = 25,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: str = "atlas-training@localhost",
        # Twilio settings (preferred over email gateway)
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_from_number: Optional[str] = None,
        # Quiet hours (don't send non-critical during these times)
        quiet_start: time = time(22, 0),  # 10pm
        quiet_end: time = time(8, 0),     # 8am
        # Feature flags
        enable_progress: bool = True,
        enable_checkpoint: bool = True,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.phone_number = phone_number.replace("-", "").replace(" ", "")
        # Format for Twilio (needs +1 prefix)
        if not self.phone_number.startswith("+"):
            self.phone_number = "+1" + self.phone_number

        self.carrier = carrier.lower()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end
        self.enable_progress = enable_progress
        self.enable_checkpoint = enable_checkpoint

        # Twilio setup (check env vars if not passed)
        self.twilio_sid = twilio_account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.twilio_token = twilio_auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_from = twilio_from_number or os.environ.get("TWILIO_FROM_NUMBER")

        # Initialize Twilio client if available
        self.twilio_client = None
        if TWILIO_AVAILABLE and self.twilio_sid and self.twilio_token and self.twilio_from:
            try:
                self.twilio_client = TwilioClient(self.twilio_sid, self.twilio_token)
                self.use_twilio = True
                print(f"[ALERTS] Using Twilio SMS to {self.phone_number}")
            except Exception as e:
                print(f"[ALERTS] Twilio init failed: {e}, falling back to email gateway")
                self.use_twilio = False
        else:
            self.use_twilio = False

        # Build email gateway address (fallback)
        gateway = self.CARRIER_GATEWAYS.get(self.carrier)
        if gateway:
            # Strip +1 for email gateway
            email_phone = self.phone_number.lstrip("+1")
            self.to_address = f"{email_phone}@{gateway}"
        else:
            self.to_address = None
            if not self.use_twilio:
                print(f"[ALERTS] Warning: Unknown carrier '{carrier}' and no Twilio configured")

        # Track last progress send to avoid spam
        self.last_progress_time: Optional[datetime] = None
        self.progress_interval_minutes = 60  # Minimum time between progress alerts

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        now = datetime.now().time()

        # Handle overnight quiet hours (e.g., 10pm - 8am)
        if self.quiet_start > self.quiet_end:
            # Quiet hours span midnight
            return now >= self.quiet_start or now < self.quiet_end
        else:
            # Quiet hours within same day
            return self.quiet_start <= now < self.quiet_end

    def _should_send(self, level: AlertLevel) -> bool:
        """Determine if alert should be sent based on level and time."""
        if not self.enabled:
            return False

        # Critical and complete always send
        if level in (AlertLevel.CRITICAL, AlertLevel.COMPLETE):
            return True

        # Check quiet hours for non-critical
        if self._is_quiet_hours():
            return False

        # Check feature flags
        if level == AlertLevel.PROGRESS and not self.enable_progress:
            return False
        if level == AlertLevel.CHECKPOINT and not self.enable_checkpoint:
            return False

        # Rate limit progress alerts
        if level == AlertLevel.PROGRESS:
            if self.last_progress_time:
                elapsed = (datetime.now() - self.last_progress_time).total_seconds() / 60
                if elapsed < self.progress_interval_minutes:
                    return False
            self.last_progress_time = datetime.now()

        return True

    def _send_twilio(self, body: str) -> bool:
        """Send SMS via Twilio API."""
        try:
            message = self.twilio_client.messages.create(
                body=body[:160],  # SMS limit
                from_=self.twilio_from,
                to=self.phone_number
            )
            print(f"[ALERT] Twilio SMS sent: {message.sid}")
            return True
        except Exception as e:
            print(f"[ALERT] Twilio failed: {e}")
            return False

    def _send_email_gateway(self, subject: str, body: str) -> bool:
        """Send SMS via email-to-SMS gateway."""
        if not self.to_address:
            print("[ALERT] No email gateway configured")
            return False

        try:
            msg = EmailMessage()
            # SMS has char limits, keep it short
            msg.set_content(body[:140])  # SMS limit
            msg['Subject'] = subject[:30]  # Short subject
            msg['From'] = self.from_email
            msg['To'] = self.to_address

            if self.smtp_port == 465:
                # SSL connection (Google Workspace SMTP Relay)
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as server:
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            elif self.smtp_user and self.smtp_password:
                # TLS with authentication (e.g., Gmail with app password)
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            else:
                # Plain SMTP (local relay, IP-authenticated)
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.send_message(msg)

            print(f"[ALERT] SMS sent via {self.smtp_host}")
            return True
        except Exception as e:
            print(f"[ALERT] Email gateway failed: {e}")
            return False

    def _send_sms(self, subject: str, body: str) -> bool:
        """Send SMS via best available method."""
        # Prefer Twilio if configured
        if self.use_twilio and self.twilio_client:
            # Include subject in body for Twilio
            full_msg = f"{subject}: {body}"
            return self._send_twilio(full_msg)

        # Fall back to email gateway
        return self._send_email_gateway(subject, body)

    def send(self, level: AlertLevel, message: str, subject: Optional[str] = None) -> bool:
        """
        Send an alert if appropriate for the level and time.

        Args:
            level: Alert severity level
            message: Alert message body
            subject: Optional subject (defaults to level name)

        Returns:
            True if alert was sent, False otherwise
        """
        if not self._should_send(level):
            return False

        # Default subject based on level
        if subject is None:
            prefix = {
                AlertLevel.CRITICAL: "ATLAS FAIL",
                AlertLevel.CHECKPOINT: "ATLAS CKPT",
                AlertLevel.PROGRESS: "ATLAS",
                AlertLevel.COMPLETE: "ATLAS DONE",
            }
            subject = prefix.get(level, "ATLAS")

        return self._send_sms(subject, message)

    # Convenience methods
    def critical(self, message: str) -> bool:
        """Send critical alert (always sends, 24/7)."""
        return self.send(AlertLevel.CRITICAL, message)

    def checkpoint(self, step: int, loss: float) -> bool:
        """Send checkpoint notification (respects quiet hours)."""
        msg = f"Step {step}: loss={loss:.4f}"
        return self.send(AlertLevel.CHECKPOINT, msg)

    def progress(self, step: int, total: int, loss: float, eta_hours: float) -> bool:
        """Send progress update (respects quiet hours, rate limited)."""
        pct = step / total * 100
        msg = f"Step {step}/{total} ({pct:.1f}%) loss={loss:.4f} ETA:{eta_hours:.1f}h"
        return self.send(AlertLevel.PROGRESS, msg)

    def complete(self, total_steps: int, final_loss: float, duration_hours: float) -> bool:
        """Send training complete notification (always sends)."""
        msg = f"Done! {total_steps} steps, loss={final_loss:.4f}, took {duration_hours:.1f}h"
        return self.send(AlertLevel.COMPLETE, msg)

    def nan_detected(self, step: int, last_good_loss: float) -> bool:
        """Send NaN detection alert (always sends)."""
        msg = f"NaN at step {step}! Last good loss: {last_good_loss:.4f}. Training stopped."
        return self.send(AlertLevel.CRITICAL, msg)

    def loss_explosion(self, step: int, prev_loss: float, curr_loss: float) -> bool:
        """Send loss explosion alert (always sends)."""
        msg = f"Loss exploded at step {step}: {prev_loss:.2f} -> {curr_loss:.2f}"
        return self.send(AlertLevel.CRITICAL, msg)


def create_alerts_from_config(config: dict) -> Optional[TrainingAlerts]:
    """
    Create TrainingAlerts from config dict.

    Config format:
        alerts:
            enabled: true
            phone: "YOUR_PHONE"
            carrier: "tmobile"
            # Twilio (recommended)
            twilio_account_sid: "ACxxxx"  # or use env TWILIO_ACCOUNT_SID
            twilio_auth_token: "xxxx"     # or use env TWILIO_AUTH_TOKEN
            twilio_from_number: "+1xxx"   # or use env TWILIO_FROM_NUMBER
            # Email gateway (fallback)
            smtp_host: "localhost"
            smtp_port: 25
            quiet_start: "22:00"
            quiet_end: "08:00"
            progress_updates: true
            checkpoint_updates: true
    """
    alerts_config = config.get("alerts", {})

    if not alerts_config.get("enabled", False):
        return None

    phone = alerts_config.get("phone")
    if not phone:
        print("[ALERT] No phone number configured, alerts disabled")
        return None

    # Parse quiet hours
    quiet_start = time(22, 0)
    quiet_end = time(8, 0)

    if "quiet_start" in alerts_config:
        h, m = map(int, alerts_config["quiet_start"].split(":"))
        quiet_start = time(h, m)
    if "quiet_end" in alerts_config:
        h, m = map(int, alerts_config["quiet_end"].split(":"))
        quiet_end = time(h, m)

    return TrainingAlerts(
        phone_number=phone,
        carrier=alerts_config.get("carrier", "tmobile"),
        smtp_host=alerts_config.get("smtp_host", "localhost"),
        smtp_port=alerts_config.get("smtp_port", 25),
        smtp_user=alerts_config.get("smtp_user"),
        smtp_password=alerts_config.get("smtp_password"),
        from_email=alerts_config.get("from_email", "atlas-training@localhost"),
        twilio_account_sid=alerts_config.get("twilio_account_sid"),
        twilio_auth_token=alerts_config.get("twilio_auth_token"),
        twilio_from_number=alerts_config.get("twilio_from_number"),
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        enable_progress=alerts_config.get("progress_updates", True),
        enable_checkpoint=alerts_config.get("checkpoint_updates", True),
        enabled=True,
    )
