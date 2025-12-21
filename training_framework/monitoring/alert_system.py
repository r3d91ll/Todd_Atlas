"""
Alert System - Telegram notifications with configurable thresholds.

Provides real-time alerts for training anomalies, checkpoints, and completion.
"""

import time
import logging
import requests
from collections import deque
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    bot_token: str
    chat_id: str
    enabled: bool = True
    quiet_hours: Tuple[int, int] = (22, 8)  # 10pm to 8am
    cooldown_seconds: int = 300  # Min time between same alerts


@dataclass
class AlertThreshold:
    """Threshold configuration for a metric."""
    metric_name: str
    threshold: float
    severity: AlertSeverity
    above: bool = True  # True = alert when above threshold
    message_template: str = "{metric_name} = {value:.4f} ({direction} {threshold})"


class TelegramNotifier:
    """
    Send notifications via Telegram Bot API.

    Setup:
    1. Create bot via @BotFather on Telegram
    2. Get bot token
    3. Get chat ID by messaging bot and checking /getUpdates
    """

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"
        self.logger = logging.getLogger(self.__class__.__name__)

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message. Returns True if successful.

        Args:
            message: Message text (supports Markdown)
            parse_mode: "Markdown" or "HTML"
        """
        if not self.config.enabled:
            self.logger.debug(f"Telegram disabled, would send: {message[:50]}...")
            return True

        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.config.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )

            if response.status_code == 200:
                return True
            else:
                self.logger.warning(
                    f"Telegram API error: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.Timeout:
            self.logger.warning("Telegram request timed out")
            return False
        except Exception as e:
            self.logger.warning(f"Telegram notification failed: {e}")
            return False

    def send_training_start(self, experiment_name: str, config_summary: str) -> bool:
        """Notify training has started."""
        msg = f"""🚀 *Training Started*

*Experiment*: `{experiment_name}`

{config_summary}

_Monitoring active_"""
        return self.send(msg)

    def send_checkpoint(self, step: int, loss: float, gate_mean: float) -> bool:
        """Notify checkpoint saved."""
        msg = f"""💾 *Checkpoint Saved*

*Step*: {step:,}
*Loss*: {loss:.4f}
*Gate Mean*: {gate_mean:.2%}"""
        return self.send(msg)

    def send_training_complete(
        self, total_steps: int, final_loss: float, final_gate: float
    ) -> bool:
        """Notify training completed."""
        msg = f"""✅ *Training Complete*

*Total Steps*: {total_steps:,}
*Final Loss*: {final_loss:.4f}
*Final Gate Mean*: {final_gate:.2%}"""
        return self.send(msg)

    def send_alert(self, message: str, severity: str = "warning") -> bool:
        """Send an alert with severity indicator."""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        emoji = emoji_map.get(severity.lower(), "⚠️")
        msg = f"{emoji} *Alert* ({severity.upper()})\n\n{message}"
        return self.send(msg)


class AlertSystem:
    """
    Monitors metrics and sends alerts when thresholds are exceeded.

    Features:
    - Configurable thresholds per metric
    - Cooldown to prevent spam
    - Quiet hours support
    - Multiple severity levels
    """

    def __init__(self, telegram_config: Optional[TelegramConfig] = None):
        self.notifier = TelegramNotifier(telegram_config) if telegram_config else None
        self.logger = logging.getLogger(self.__class__.__name__)

        # Thresholds: metric_name -> AlertThreshold
        self._thresholds: Dict[str, AlertThreshold] = {}

        # Cooldown tracking: metric_name -> last_alert_time
        self._cooldowns: Dict[str, float] = {}

        # Alert history for dashboard (bounded to prevent unbounded growth)
        self._alert_history: deque = deque(maxlen=1000)

        # Default cooldown
        self._default_cooldown = telegram_config.cooldown_seconds if telegram_config else 300

    def register_threshold(
        self,
        metric_name: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        above: bool = True,
        message_template: Optional[str] = None,
    ) -> None:
        """Register an alert threshold for a metric."""
        self._thresholds[metric_name] = AlertThreshold(
            metric_name=metric_name,
            threshold=threshold,
            severity=severity,
            above=above,
            message_template=message_template or AlertThreshold.message_template,
        )

    def register_thresholds_from_adapter(
        self, thresholds: Dict[str, Tuple[float, str]]
    ) -> None:
        """
        Register thresholds from adapter format.

        Args:
            thresholds: Dict of metric_name -> (threshold_value, severity_string)
        """
        for metric_name, (threshold, severity_str) in thresholds.items():
            # Parse severity and direction
            severity_str = severity_str.lower()
            above = not severity_str.endswith('_below')
            if severity_str.endswith('_below'):
                severity_str = severity_str[:-6]

            severity = AlertSeverity(severity_str) if severity_str in ['info', 'warning', 'critical'] else AlertSeverity.WARNING

            self.register_threshold(
                metric_name=metric_name,
                threshold=threshold,
                severity=severity,
                above=above,
            )

    def check_thresholds(self, metrics: Dict[str, Any]) -> list:
        """
        Check all registered thresholds against current metrics.

        Returns list of triggered alerts.
        """
        triggered = []

        for metric_name, threshold in self._thresholds.items():
            value = metrics.get(metric_name)
            if value is None:
                continue

            # Check if threshold is exceeded
            exceeded = (
                (threshold.above and value > threshold.threshold) or
                (not threshold.above and value < threshold.threshold)
            )

            if exceeded and self._should_alert(metric_name):
                direction = "above" if threshold.above else "below"
                message = threshold.message_template.format(
                    metric_name=metric_name,
                    value=value,
                    direction=direction,
                    threshold=threshold.threshold,
                )

                # Send alert
                if self.notifier:
                    self.notifier.send_alert(message, threshold.severity.value)

                # Record alert
                alert_record = {
                    'timestamp': time.time(),
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold.threshold,
                    'severity': threshold.severity.value,
                    'message': message,
                }
                self._alert_history.append(alert_record)
                triggered.append(alert_record)

                # Update cooldown
                self._cooldowns[metric_name] = time.time()

                self.logger.warning(f"Alert: {message}")

        return triggered

    def _should_alert(self, metric_name: str) -> bool:
        """Check if enough time has passed since last alert for this metric."""
        last_alert = self._cooldowns.get(metric_name, 0)
        return (time.time() - last_alert) >= self._default_cooldown

    def get_alert_history(self, limit: int = 100) -> list:
        """Get recent alert history."""
        history_list = list(self._alert_history)
        return history_list[-limit:]

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()

    # Convenience methods that delegate to notifier
    def send_training_start(self, experiment_name: str, config_summary: str) -> bool:
        if self.notifier:
            return self.notifier.send_training_start(experiment_name, config_summary)
        return True

    def send_checkpoint(self, step: int, loss: float, gate_mean: float) -> bool:
        if self.notifier:
            return self.notifier.send_checkpoint(step, loss, gate_mean)
        return True

    def send_training_complete(
        self, total_steps: int, final_loss: float, final_gate: float
    ) -> bool:
        if self.notifier:
            return self.notifier.send_training_complete(total_steps, final_loss, final_gate)
        return True

    def send_alert(self, message: str, severity: str = "warning") -> bool:
        if self.notifier:
            return self.notifier.send_alert(message, severity)
        return True
