# tts_notification.py
"""Voice-based notification system using TTSClient."""

from typing import List, Dict, Optional, Union, Any, Literal, Callable
from enum import Enum
from pathlib import Path
import os
import time
import logging
from pydantic import BaseModel, Field
import json

from .tts_module import TTSClient, GenerateAudioRequest, play_audio, get_output_path
from .models.task_models import Task, TaskStatus, TaskPriority, Epic
from .utils.fs_utils import resolve_path, append_to_file, safe_save_json, load_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationPriority(str, Enum):
    """Priority levels for voice notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationType(str, Enum):
    """Types of voice notifications."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    MILESTONE = "milestone"
    PROGRESS = "progress"

class VoiceProfile(BaseModel):
    """Configuration for a voice profile."""
    voice_id: str = "af_nicole"
    speed: float = Field(1.0, ge=0.1, le=2.0)
    auto_play: bool = True
    
    # Optional path to save audio files
    output_dir: Optional[str] = "artifacts/audio"
    
    # Text formatting templates
    templates: Dict[NotificationType, str] = Field(
        default_factory=lambda: {
            NotificationType.INFO: "Information: {message}",
            NotificationType.SUCCESS: "Success! {message}",
            NotificationType.WARNING: "Warning: {message}",
            NotificationType.ERROR: "Error: {message}",
            NotificationType.MILESTONE: "Milestone achieved: {message}",
            NotificationType.PROGRESS: "Progress update: {message}"
        }
    )

class NotificationConfig(BaseModel):
    """Configuration for the notification system."""
    enabled: bool = True
    
    # Default voice profiles for different notification priorities
    voices: Dict[NotificationPriority, VoiceProfile] = Field(
        default_factory=lambda: {
            NotificationPriority.LOW: VoiceProfile(voice_id="af_bella", speed=1.0),
            NotificationPriority.MEDIUM: VoiceProfile(voice_id="af_bella", speed=1.0),
            NotificationPriority.HIGH: VoiceProfile(voice_id="af_bella", speed=0.9),
            NotificationPriority.CRITICAL: VoiceProfile(voice_id="af_bella", speed=0.8),
        }
    )
    
    # Notification history
    history_enabled: bool = True
    max_history_items: int = 100
    history_file: str = "artifacts/audio/notification_history.json"
    
    # Rate limiting
    min_interval_seconds: float = 5.0  # Minimum time between notifications
    
    # Silent periods
    silent_mode: bool = False

class NotificationHistory(BaseModel):
    """History of notifications sent."""
    notifications: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add(self, notification_data: Dict[str, Any]) -> None:
        """Add a notification to history."""
        self.notifications.append(notification_data)
        # Trim if exceeding max length
        if len(self.notifications) > 100:  # Hardcoded for safety
            self.notifications = self.notifications[-100:]

class NotificationManager:
    """Manages voice notifications using TTS."""
    
    def __init__(
        self,
        tts_base_url: str = "http://127.0.0.1:7860",
        config: Optional[NotificationConfig] = None
    ):
        """
        Initialize the notification manager.
        
        Args:
            tts_base_url: URL for the TTS API
            config: Configuration for notifications
        """
        self.tts_client = TTSClient(base_url=tts_base_url)
        self.config = config or NotificationConfig()
        self.history = NotificationHistory()
        self.last_notification_time = 0
        
        # Load history if available
        self._load_history()
    
    def _load_history(self) -> None:
        """Load notification history from file."""
        if self.config.history_enabled:
            history_path = resolve_path(self.config.history_file)
            if os.path.exists(history_path):
                try:
                    history_data = load_json(history_path)
                    self.history = NotificationHistory.model_validate(history_data)
                except Exception as e:
                    logger.error(f"Error loading notification history: {e}")
    
    def _save_history(self) -> None:
        """Save notification history to file."""
        if self.config.history_enabled:
            try:
                # Use direct file writing instead of safe_save_json
                history_path = resolve_path(self.config.history_file, create_parents=True)
                
                # Convert to dictionary first
                history_data = self.history.model_dump()
                
                # Write directly to file
                with open(history_path, 'w') as f:
                    json.dump(history_data, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error saving notification history: {e}")
    
    def notify(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        auto_play: Optional[bool] = None,
        save_to_file: bool = True,
        filename: Optional[str] = None,
        voice_profile: Optional[VoiceProfile] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Send a voice notification.
        
        Args:
            message: The message to speak
            notification_type: Type of notification
            priority: Priority level
            auto_play: Whether to play the audio immediately
            save_to_file: Whether to save the audio to a file
            filename: Optional filename to use
            voice_profile: Optional voice profile override
            metadata: Additional metadata to store with the notification
            
        Returns:
            Path to the generated audio file or None
        """
        # Check if notifications are enabled
        if not self.config.enabled or self.config.silent_mode:
            logger.info(f"Notification suppressed (silent mode): {message}")
            return None
        
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_notification_time < self.config.min_interval_seconds:
            logger.info(f"Notification rate-limited: {message}")
            return None
        
        # Get the appropriate voice profile
        profile = voice_profile or self.config.voices.get(priority, self.config.voices[NotificationPriority.MEDIUM])
        
        # Apply message template
        template = profile.templates.get(notification_type, "{message}")
        formatted_message = template.format(message=message)
        
        # Determine output path if saving to file
        output_path = None
        if save_to_file:
            if filename:
                output_path = resolve_path(f"{profile.output_dir}/{filename}", create_parents=True)
            else:
                timestamp = int(time.time())
                output_path = resolve_path(
                    f"{profile.output_dir}/{notification_type.value}_{timestamp}.wav", 
                    create_parents=True
                )
        
        # Determine whether to auto-play
        should_play = auto_play if auto_play is not None else profile.auto_play
        
        try:
            # Create TTS request
            request = GenerateAudioRequest(
                text=formatted_message,
                voice=profile.voice_id,
                speed=profile.speed
            )
            
            # Generate audio
            result = self.tts_client.generate_audio(request, save_to=str(output_path) if output_path else None)
            
            # Play if requested
            if should_play and "audio_path" in result:
                play_audio(result["audio_path"])
            
            # Update last notification time
            self.last_notification_time = current_time
            
            # Record in history if enabled
            if self.config.history_enabled:
                notification_data = {
                    "timestamp": current_time,
                    "message": message,
                    "formatted_message": formatted_message,
                    "type": notification_type.value,
                    "priority": priority.value,
                    "voice": profile.voice_id,
                    "audio_path": result.get("audio_path", ""),
                    "metadata": metadata or {}
                }
                self.history.add(notification_data)
                self._save_history()
            
            return result.get("audio_path")
        
        except Exception as e:
            logger.error(f"Error generating notification: {e}")
            return None
    
    def info(self, message: str, **kwargs) -> Optional[str]:
        """Send an informational notification."""
        return self.notify(message, NotificationType.INFO, NotificationPriority.LOW, **kwargs)
    
    def success(self, message: str, **kwargs) -> Optional[str]:
        """Send a success notification."""
        return self.notify(message, NotificationType.SUCCESS, NotificationPriority.MEDIUM, **kwargs)
    
    def warning(self, message: str, **kwargs) -> Optional[str]:
        """Send a warning notification."""
        return self.notify(message, NotificationType.WARNING, NotificationPriority.HIGH, **kwargs)
    
    def error(self, message: str, **kwargs) -> Optional[str]:
        """Send an error notification."""
        return self.notify(message, NotificationType.ERROR, NotificationPriority.CRITICAL, **kwargs)
    
    def milestone(self, message: str, **kwargs) -> Optional[str]:
        """Send a milestone notification."""
        return self.notify(message, NotificationType.MILESTONE, NotificationPriority.HIGH, **kwargs)
    
    def progress(self, message: str, **kwargs) -> Optional[str]:
        """Send a progress update notification."""
        return self.notify(message, NotificationType.PROGRESS, NotificationPriority.LOW, **kwargs)
    
    def task_status_update(self, task: Task, old_status: Optional[TaskStatus] = None) -> Optional[str]:
        """
        Notify about a task status change.
        
        Args:
            task: The task that changed status
            old_status: The previous status (if any)
            
        Returns:
            Path to the generated audio file or None
        """
        if old_status is None or old_status == task.status:
            message = f"Task {task.title} is {task.status.value}."
        else:
            message = f"Task {task.title} changed from {old_status.value} to {task.status.value}."
        
        # Map task status to notification type
        notification_type = NotificationType.INFO
        if task.status == TaskStatus.COMPLETED:
            notification_type = NotificationType.SUCCESS
        elif task.status == TaskStatus.BLOCKED:
            notification_type = NotificationType.WARNING
        elif task.status == TaskStatus.FAILED:
            notification_type = NotificationType.ERROR
        
        # Map task priority to notification priority
        notification_priority = NotificationPriority.MEDIUM
        if task.priority == TaskPriority.CRITICAL:
            notification_priority = NotificationPriority.CRITICAL
        elif task.priority == TaskPriority.HIGH:
            notification_priority = NotificationPriority.HIGH
        elif task.priority == TaskPriority.LOW:
            notification_priority = NotificationPriority.LOW
        
        # Include task metadata
        metadata = {
            "task_id": str(task.id),
            "task_title": task.title,
            "old_status": old_status.value if old_status else None,
            "new_status": task.status.value
        }
        
        return self.notify(
            message=message,
            notification_type=notification_type,
            priority=notification_priority,
            metadata=metadata
        )
    
    def epic_progress(self, epic: Epic, progress: float) -> Optional[str]:
        """
        Notify about epic progress.
        
        Args:
            epic: The epic to report on
            progress: Current progress percentage
            
        Returns:
            Path to the generated audio file or None
        """
        # Format progress as percentage
        progress_pct = round(progress)
        
        # Create appropriate message based on progress
        if progress_pct == 100:
            message = f"Epic {epic.title} is now complete!"
            notification_type = NotificationType.SUCCESS
            priority = NotificationPriority.HIGH
        elif progress_pct >= 75:
            message = f"Epic {epic.title} is {progress_pct}% complete. Getting close to completion."
            notification_type = NotificationType.PROGRESS
            priority = NotificationPriority.MEDIUM
        elif progress_pct >= 50:
            message = f"Epic {epic.title} is {progress_pct}% complete. Making good progress."
            notification_type = NotificationType.PROGRESS
            priority = NotificationPriority.MEDIUM
        elif progress_pct >= 25:
            message = f"Epic {epic.title} is {progress_pct}% complete. Moving forward."
            notification_type = NotificationType.PROGRESS
            priority = NotificationPriority.LOW
        else:
            message = f"Epic {epic.title} is {progress_pct}% complete. Just getting started."
            notification_type = NotificationType.PROGRESS
            priority = NotificationPriority.LOW
        
        # Include epic metadata
        metadata = {
            "epic_id": str(epic.id),
            "epic_title": epic.title,
            "progress": progress_pct
        }
        
        return self.notify(
            message=message,
            notification_type=notification_type,
            priority=priority,
            metadata=metadata
        )
    
    def silence(self, enable: bool = True) -> None:
        """
        Enable or disable silent mode.
        
        Args:
            enable: Whether to enable silent mode
        """
        self.config.silent_mode = enable
        logger.info(f"Silent mode {'enabled' if enable else 'disabled'}")
    
    def get_history(self, limit: int = 10, notification_type: Optional[NotificationType] = None) -> List[Dict[str, Any]]:
        """
        Get recent notification history.
        
        Args:
            limit: Maximum number of items to return
            notification_type: Optional filter by notification type
            
        Returns:
            List of notification history items
        """
        if not self.config.history_enabled:
            return []
        
        # Filter by type if specified
        if notification_type:
            filtered = [n for n in self.history.notifications if n.get("type") == notification_type.value]
        else:
            filtered = self.history.notifications
        
        # Return most recent first, limited by count
        return sorted(filtered, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]