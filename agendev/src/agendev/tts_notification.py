"""Voice-based notification system using TTSClient."""
from typing import Dict, List, Optional, Union, Any
import os
import time
import threading
import logging
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from .tts_module import TTSClient, GenerateAudioRequest, play_audio, get_available_voices
from .utils.fs_utils import get_audio_directory, ensure_directory, save_json, load_json
from .models.task_models import Task, TaskStatus, TaskPriority, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_notification")


class NotificationPriority(str, Enum):
    """Priority level for notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(str, Enum):
    """Types of notifications."""
    TASK_STATUS = "task_status"
    MILESTONE = "milestone"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class Notification:
    """Represents a single notification."""
    def __init__(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        task_id: Optional[UUID] = None,
        related_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a notification.
        
        Args:
            message: Notification message
            notification_type: Type of notification
            priority: Priority level
            task_id: ID of the related task (if any)
            related_data: Additional data related to the notification
        """
        self.id = uuid4()
        self.message = message
        self.notification_type = notification_type
        self.priority = priority
        self.task_id = task_id
        self.related_data = related_data or {}
        self.timestamp = datetime.now()
        self.audio_path = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to a dictionary."""
        return {
            "id": str(self.id),
            "message": self.message,
            "notification_type": self.notification_type.value,
            "priority": self.priority.value,
            "task_id": str(self.task_id) if self.task_id else None,
            "related_data": self.related_data,
            "timestamp": self.timestamp.isoformat(),
            "audio_path": str(self.audio_path) if self.audio_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Notification':
        """Create a notification from a dictionary."""
        notification = cls(
            message=data["message"],
            notification_type=NotificationType(data["notification_type"]),
            priority=NotificationPriority(data["priority"]),
            task_id=UUID(data["task_id"]) if data.get("task_id") else None,
            related_data=data.get("related_data", {})
        )
        notification.id = UUID(data["id"])
        notification.timestamp = datetime.fromisoformat(data["timestamp"])
        notification.audio_path = Path(data["audio_path"]) if data.get("audio_path") else None
        return notification


class TTSNotificationSystem:
    """Voice-based notification system using TTSClient."""
    
    def __init__(
        self,
        tts_client: Optional[TTSClient] = None,
        default_voice: str = "af_nicole",
        notification_history_limit: int = 100,
        auto_play: bool = True
    ):
        """
        Initialize the TTS notification system.
        
        Args:
            tts_client: TTS client for generating audio
            default_voice: Default voice to use
            notification_history_limit: Maximum number of notifications to keep in history
            auto_play: Whether to automatically play notifications
        """
        self.tts_client = tts_client or TTSClient()
        self.default_voice = default_voice
        self.notification_history_limit = notification_history_limit
        self.auto_play = auto_play
        
        # Voice selection for different notification types
        self.voice_mapping = {
            NotificationType.TASK_STATUS: "af_nicole",
            NotificationType.MILESTONE: "af_heart",
            NotificationType.ERROR: "af_bella",
            NotificationType.WARNING: "am_echo",
            NotificationType.INFO: "af_nicole",
            NotificationType.SUCCESS: "af_sky"
        }
        
        # Speed settings for different priorities
        self.speed_mapping = {
            NotificationPriority.LOW: 0.9,
            NotificationPriority.MEDIUM: 1.0,
            NotificationPriority.HIGH: 1.1,
            NotificationPriority.CRITICAL: 1.2
        }
        
        # Notification history
        self.notification_history: List[Notification] = []
        
        # Load notification history
        self._load_notification_history()
    
    def _load_notification_history(self) -> None:
        """Load notification history from disk."""
        audio_dir = get_audio_directory()
        history_path = audio_dir / "notification_history.json"
        
        if history_path.exists():
            history_data = load_json(history_path)
            if history_data and "notifications" in history_data:
                self.notification_history = [
                    Notification.from_dict(item) for item in history_data["notifications"]
                ]
    
    def _save_notification_history(self) -> None:
        """Save notification history to disk."""
        audio_dir = get_audio_directory()
        history_path = audio_dir / "notification_history.json"
        
        # Limit the number of notifications in history
        if len(self.notification_history) > self.notification_history_limit:
            self.notification_history = self.notification_history[-self.notification_history_limit:]
        
        history_data = {
            "notifications": [n.to_dict() for n in self.notification_history]
        }
        
        save_json(history_data, history_path)
    
    def add_notification(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        task_id: Optional[UUID] = None,
        related_data: Optional[Dict[str, Any]] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Add a notification and generate audio.
        
        Args:
            message: Notification message
            notification_type: Type of notification
            priority: Priority level
            task_id: ID of the related task (if any)
            related_data: Additional data related to the notification
            voice: Voice to use (overrides default mapping)
            speed: Speed to use (overrides default mapping)
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        # Create notification
        notification = Notification(
            message=message,
            notification_type=notification_type,
            priority=priority,
            task_id=task_id,
            related_data=related_data
        )
        
        # Generate audio
        try:
            # Select voice and speed
            selected_voice = voice or self.voice_mapping.get(notification_type, self.default_voice)
            selected_speed = speed or self.speed_mapping.get(priority, 1.0)
            
            # Determine auto_play behavior
            play_now = auto_play if auto_play is not None else self.auto_play
            
            # Generate audio filename
            audio_dir = get_audio_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"notification_{notification_type.value}_{timestamp}_{str(notification.id)[:8]}.wav"
            audio_path = audio_dir / filename
            
            # Generate audio request
            request = GenerateAudioRequest(
                text=message,
                voice=selected_voice,
                speed=selected_speed
            )
            
            # Generate audio
            result = self.tts_client.generate_audio(request, save_to=str(audio_path))
            
            # Set audio path
            if "audio_path" in result and os.path.exists(result["audio_path"]):
                notification.audio_path = Path(result["audio_path"])
                
                # Play audio if requested
                if play_now:
                    play_audio(result["audio_path"])
        
        except Exception as e:
            logger.error(f"Error generating audio notification: {str(e)}")
        
        # Add to history
        self.notification_history.append(notification)
        self._save_notification_history()
        
        return notification
    
    def notify_task_status(
        self,
        task: Task,
        old_status: Optional[TaskStatus] = None,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Create a notification for a task status change.
        
        Args:
            task: The task being updated
            old_status: Previous task status (if any)
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        # Determine notification priority based on task priority
        priority_mapping = {
            TaskPriority.LOW: NotificationPriority.LOW,
            TaskPriority.MEDIUM: NotificationPriority.MEDIUM,
            TaskPriority.HIGH: NotificationPriority.HIGH,
            TaskPriority.CRITICAL: NotificationPriority.CRITICAL
        }
        notification_priority = priority_mapping.get(task.priority, NotificationPriority.MEDIUM)
        
        # Create message based on status
        if old_status and old_status != task.status:
            message = f"Task '{task.title}' has changed from {old_status.value} to {task.status.value}."
        else:
            message = f"Task '{task.title}' is now {task.status.value}."
        
        # Add extra context for certain statuses
        if task.status == TaskStatus.COMPLETED:
            message += f" This task took approximately {task.actual_hours:.1f} hours to complete."
        elif task.status == TaskStatus.BLOCKED:
            message += " This task is currently blocked by dependencies."
        elif task.status == TaskStatus.FAILED:
            message += " This task has encountered issues and could not be completed."
        
        return self.add_notification(
            message=message,
            notification_type=NotificationType.TASK_STATUS,
            priority=notification_priority,
            task_id=task.id,
            related_data={"task_status": task.status.value},
            auto_play=auto_play
        )
    
    def notify_milestone(
        self,
        title: str,
        description: str,
        priority: NotificationPriority = NotificationPriority.HIGH,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Create a notification for a milestone achievement.
        
        Args:
            title: Milestone title
            description: Milestone description
            priority: Notification priority
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        message = f"Milestone achieved: {title}. {description}"
        
        return self.add_notification(
            message=message,
            notification_type=NotificationType.MILESTONE,
            priority=priority,
            auto_play=auto_play
        )
    
    def notify_error(
        self,
        error_message: str,
        details: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.HIGH,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Create an error notification.
        
        Args:
            error_message: Error message
            details: Additional error details
            priority: Notification priority
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        message = f"Error: {error_message}"
        if details:
            message += f". {details}"
        
        return self.add_notification(
            message=message,
            notification_type=NotificationType.ERROR,
            priority=priority,
            auto_play=auto_play
        )
    
    def notify_warning(
        self,
        warning_message: str,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Create a warning notification.
        
        Args:
            warning_message: Warning message
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        return self.add_notification(
            message=f"Warning: {warning_message}",
            notification_type=NotificationType.WARNING,
            priority=NotificationPriority.MEDIUM,
            auto_play=auto_play
        )
    
    def notify_success(
        self,
        success_message: str,
        auto_play: Optional[bool] = None
    ) -> Notification:
        """
        Create a success notification.
        
        Args:
            success_message: Success message
            auto_play: Whether to automatically play the notification
            
        Returns:
            The created notification
        """
        return self.add_notification(
            message=f"Success: {success_message}",
            notification_type=NotificationType.SUCCESS,
            priority=NotificationPriority.MEDIUM,
            auto_play=auto_play
        )
    
    def play_notification(self, notification: Union[Notification, UUID]) -> bool:
        """
        Play a notification's audio.
        
        Args:
            notification: Notification or notification ID
            
        Returns:
            True if played successfully, False otherwise
        """
        # Get notification object
        if isinstance(notification, UUID):
            notification_obj = next((n for n in self.notification_history if n.id == notification), None)
            if notification_obj is None:
                return False
        else:
            notification_obj = notification
        
        # Check if audio path exists
        if notification_obj.audio_path and notification_obj.audio_path.exists():
            try:
                play_audio(str(notification_obj.audio_path))
                return True
            except Exception as e:
                logger.error(f"Error playing notification audio: {str(e)}")
                return False
        
        return False
    
    def summarize_recent_activity(
        self,
        count: int = 5,
        auto_play: bool = True
    ) -> Optional[Notification]:
        """
        Create a summary of recent activity.
        
        Args:
            count: Number of recent notifications to summarize
            auto_play: Whether to automatically play the summary
            
        Returns:
            The created summary notification, or None if no activity
        """
        recent = self.notification_history[-count:] if len(self.notification_history) > 0 else []
        
        if not recent:
            return None
        
        # Create summary message
        message = f"Recent activity summary. Here are the last {len(recent)} notifications:\n"
        
        for i, notification in enumerate(recent, 1):
            # Format timestamp
            time_str = notification.timestamp.strftime("%H:%M:%S")
            
            # Add to message
            message += f"{i}. At {time_str}: {notification.message}\n"
        
        # Create and return summary notification
        return self.add_notification(
            message=message,
            notification_type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            auto_play=auto_play
        )
    
    def get_notification_history(
        self,
        limit: Optional[int] = None,
        notification_type: Optional[NotificationType] = None,
        min_priority: Optional[NotificationPriority] = None
    ) -> List[Notification]:
        """
        Get notification history filtered by criteria.
        
        Args:
            limit: Maximum number of notifications to return
            notification_type: Filter by notification type
            min_priority: Filter by minimum priority
            
        Returns:
            List of matching notifications
        """
        # Filter by notification type
        if notification_type:
            filtered = [n for n in self.notification_history if n.notification_type == notification_type]
        else:
            filtered = self.notification_history.copy()
        
        # Filter by minimum priority
        if min_priority:
            priority_values = {p.value: i for i, p in enumerate(NotificationPriority)}
            min_priority_value = priority_values.get(min_priority.value, 0)
            filtered = [n for n in filtered if priority_values.get(n.priority.value, 0) >= min_priority_value]
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda n: n.timestamp, reverse=True)
        
        # Apply limit
        if limit and limit > 0:
            filtered = filtered[:limit]
        
        return filtered