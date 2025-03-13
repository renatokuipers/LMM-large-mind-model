# tts_notification.py
"""Voice-based notification system using TTSClient."""

from typing import List, Dict, Optional, Union, Any, Literal, Callable, Set
from enum import Enum
from pathlib import Path
import os
import time
import logging
import threading
import json
from datetime import datetime
from pydantic import BaseModel, Field

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
    TASK_STATUS = "task_status"
    EPIC_STATUS = "epic_status"
    CODE_QUALITY = "code_quality"
    PLANNING = "planning"

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
            NotificationType.PROGRESS: "Progress update: {message}",
            NotificationType.TASK_STATUS: "Task update: {message}",
            NotificationType.EPIC_STATUS: "Epic update: {message}",
            NotificationType.CODE_QUALITY: "Code quality: {message}",
            NotificationType.PLANNING: "Planning: {message}"
        }
    )

class UserPreferences(BaseModel):
    """User preferences for notifications."""
    enabled_notification_types: Set[NotificationType] = Field(
        default_factory=lambda: set(NotificationType)
    )
    minimum_priority: NotificationPriority = NotificationPriority.LOW
    auto_play: bool = True
    voice_id: Optional[str] = None
    speed: Optional[float] = None
    notification_volume: float = Field(1.0, ge=0.0, le=1.0)

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
    min_interval_seconds: float = 1.0  # Minimum time between notifications
    
    # Priority thresholds - notifications below this priority are suppressed
    minimum_priority: NotificationPriority = NotificationPriority.LOW
    
    # Silent periods
    silent_mode: bool = False
    
    # User preferences
    user_preferences: Optional[UserPreferences] = None
    
    # Default voice settings
    default_voice_id: str = "af_bella"
    default_speed: float = 1.0
    
    # Notification queue settings
    max_queue_size: int = 10
    queue_processing_interval: float = 1.0  # seconds

class NotificationHistory(BaseModel):
    """History of notifications sent."""
    notifications: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add(self, notification_data: Dict[str, Any]) -> None:
        """Add a notification to history."""
        self.notifications.append(notification_data)
        # Trim if exceeding max length
        if len(self.notifications) > 100:  # Hardcoded for safety
            self.notifications = self.notifications[-100:]

class QueuedNotification(BaseModel):
    """A notification in the queue waiting to be processed."""
    message: str
    notification_type: NotificationType
    priority: NotificationPriority
    auto_play: Optional[bool] = None
    save_to_file: bool = True
    filename: Optional[str] = None
    voice_profile: Optional[VoiceProfile] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)
    
    class Config:
        arbitrary_types_allowed = True

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
        
        # Initialize notification queue
        self.notification_queue: List[QueuedNotification] = []
        self.queue_lock = threading.Lock()
        self.queue_processing_thread = None
        self.queue_processing_active = False
        
        # Connection status
        self.tts_available = self._check_tts_connection()
        
        # Load history if available
        self._load_history()
        
        # Start queue processing
        self._start_queue_processing()
    
    def _check_tts_connection(self) -> bool:
        """Check if TTS service is available."""
        try:
            self.tts_client.session.get(f"{self.tts_client.base_url}", timeout=5)
            logger.info(f"TTS service connection established at {self.tts_client.base_url}")
            return True
        except Exception as e:
            logger.warning(f"TTS service unavailable at {self.tts_client.base_url}: {e}")
            return False
    
    def _load_history(self) -> None:
        """Load notification history from file."""
        if self.config.history_enabled:
            history_path = resolve_path(self.config.history_file)
            if os.path.exists(history_path):
                try:
                    history_data = load_json(history_path)
                    self.history = NotificationHistory.model_validate(history_data)
                    logger.info(f"Loaded {len(self.history.notifications)} notification history items")
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
                logger.debug(f"Saved notification history to {history_path}")
            except Exception as e:
                logger.error(f"Error saving notification history: {e}")
    
    def _start_queue_processing(self) -> None:
        """Start the notification queue processing thread."""
        if not self.queue_processing_active:
            self.queue_processing_active = True
            self.queue_processing_thread = threading.Thread(
                target=self._process_notification_queue,
                daemon=True
            )
            self.queue_processing_thread.start()
            logger.info("Started notification queue processing thread")
    
    def _process_notification_queue(self) -> None:
        """Process notifications in the queue."""
        while self.queue_processing_active:
            notification_to_process = None
            
            # Get the highest priority notification from the queue
            with self.queue_lock:
                if self.notification_queue:
                    # Sort by priority and then by timestamp
                    self.notification_queue.sort(
                        key=lambda n: (
                            # Order: CRITICAL(3), HIGH(2), MEDIUM(1), LOW(0)
                            {"low": 0, "medium": 1, "high": 2, "critical": 3}[n.priority],
                            -n.timestamp  # Newer notifications first if same priority
                        ),
                        reverse=True
                    )
                    notification_to_process = self.notification_queue.pop(0)
            
            if notification_to_process:
                # Process the notification
                try:
                    self._process_single_notification(notification_to_process)
                except Exception as e:
                    logger.error(f"Error processing notification: {e}")
            
            # Sleep before checking the queue again
            time.sleep(self.config.queue_processing_interval)
    
    def _process_single_notification(self, notification: QueuedNotification) -> Optional[str]:
        """
        Process a single notification from the queue.
        
        Args:
            notification: The notification to process
            
        Returns:
            Path to the generated audio file or None
        """
        # Check if notifications are enabled
        if not self.config.enabled or self.config.silent_mode:
            logger.info(f"Notification suppressed (silent mode): {notification.message}")
            return None
        
        # Check if TTS service is available
        if not self.tts_available:
            # Periodically retry the connection
            current_time = time.time()
            if current_time - self.last_notification_time > 30.0:  # Retry every 30 seconds
                self.tts_available = self._check_tts_connection()
                self.last_notification_time = current_time
            
            if not self.tts_available:
                logger.warning(f"TTS service unavailable, notification suppressed: {notification.message}")
                return None
        
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_notification_time < self.config.min_interval_seconds:
            logger.info(f"Notification rate-limited: {notification.message}")
            return None
        
        # Check priority threshold
        priority_values = {
            NotificationPriority.LOW: 0,
            NotificationPriority.MEDIUM: 1,
            NotificationPriority.HIGH: 2,
            NotificationPriority.CRITICAL: 3
        }
        
        minimum_priority_value = priority_values.get(self.config.minimum_priority, 0)
        notification_priority_value = priority_values.get(notification.priority, 0)
        
        if notification_priority_value < minimum_priority_value:
            logger.info(f"Notification suppressed (below priority threshold): {notification.message}")
            return None
        
        # Apply user preferences if available
        if self.config.user_preferences:
            # Check if the notification type is enabled
            if notification.notification_type not in self.config.user_preferences.enabled_notification_types:
                logger.info(f"Notification suppressed (disabled by user): {notification.message}")
                return None
            
            # Check user's minimum priority
            user_min_priority_value = priority_values.get(self.config.user_preferences.minimum_priority, 0)
            if notification_priority_value < user_min_priority_value:
                logger.info(f"Notification suppressed (below user priority threshold): {notification.message}")
                return None
        
        # Get the appropriate voice profile
        profile = notification.voice_profile or self.config.voices.get(
            notification.priority, 
            self.config.voices[NotificationPriority.MEDIUM]
        )
        
        # Apply user voice preferences if available
        if self.config.user_preferences:
            if self.config.user_preferences.voice_id:
                profile.voice_id = self.config.user_preferences.voice_id
            if self.config.user_preferences.speed:
                profile.speed = self.config.user_preferences.speed
        
        # Apply message template
        template = profile.templates.get(notification.notification_type, "{message}")
        formatted_message = template.format(message=notification.message)
        
        # Determine output path if saving to file
        output_path = None
        if notification.save_to_file:
            if notification.filename:
                output_path = resolve_path(f"{profile.output_dir}/{notification.filename}", create_parents=True)
            else:
                timestamp = int(time.time())
                output_path = resolve_path(
                    f"{profile.output_dir}/{notification.notification_type.value}_{timestamp}.wav", 
                    create_parents=True
                )
        
        # Determine whether to auto-play
        should_play = notification.auto_play
        if should_play is None:
            should_play = profile.auto_play
            if self.config.user_preferences:
                should_play = self.config.user_preferences.auto_play
        
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
                    "message": notification.message,
                    "formatted_message": formatted_message,
                    "type": notification.notification_type.value,
                    "priority": notification.priority.value,
                    "voice": profile.voice_id,
                    "audio_path": result.get("audio_path", ""),
                    "metadata": notification.metadata or {}
                }
                self.history.add(notification_data)
                self._save_history()
            
            return result.get("audio_path")
        
        except Exception as e:
            logger.error(f"Error generating notification: {e}")
            # Retry TTS connection on next notification
            self.tts_available = False
            return None
    
    def notify(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        auto_play: Optional[bool] = None,
        save_to_file: bool = True,
        filename: Optional[str] = None,
        voice_profile: Optional[VoiceProfile] = None,
        metadata: Optional[Dict[str, Any]] = None,
        queued: bool = True
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
            queued: Whether to add to the queue or process immediately
            
        Returns:
            Path to the generated audio file or None if queued
        """
        # Create a queued notification object
        notification = QueuedNotification(
            message=message,
            notification_type=notification_type,
            priority=priority,
            auto_play=auto_play,
            save_to_file=save_to_file,
            filename=filename,
            voice_profile=voice_profile,
            metadata=metadata
        )
        
        # If queued mode, add to the queue and return
        if queued:
            with self.queue_lock:
                # Check if queue is full
                if len(self.notification_queue) >= self.config.max_queue_size:
                    # Remove the lowest priority notification
                    self.notification_queue.sort(
                        key=lambda n: (
                            {"low": 0, "medium": 1, "high": 2, "critical": 3}[n.priority],
                            -n.timestamp
                        )
                    )
                    self.notification_queue.pop(0)  # Remove lowest priority
                
                # Add the notification to the queue
                self.notification_queue.append(notification)
                logger.debug(f"Added notification to queue: {message[:30]}...")
            
            return None
        else:
            # Process immediately
            return self._process_single_notification(notification)
    
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
    
    def code_quality(self, message: str, priority: NotificationPriority = NotificationPriority.MEDIUM, **kwargs) -> Optional[str]:
        """Send a code quality notification."""
        return self.notify(
            message, 
            NotificationType.CODE_QUALITY, 
            priority,
            **kwargs
        )
    
    def planning(self, message: str, **kwargs) -> Optional[str]:
        """Send a planning notification."""
        return self.notify(message, NotificationType.PLANNING, NotificationPriority.MEDIUM, **kwargs)
    
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
        notification_type = NotificationType.TASK_STATUS
        
        # Map task status to notification priority
        notification_priority = NotificationPriority.MEDIUM
        if task.status == TaskStatus.COMPLETED:
            notification_priority = NotificationPriority.MEDIUM
            notification_type = NotificationType.SUCCESS
        elif task.status == TaskStatus.BLOCKED:
            notification_priority = NotificationPriority.HIGH
            notification_type = NotificationType.WARNING
        elif task.status == TaskStatus.FAILED:
            notification_priority = NotificationPriority.HIGH
            notification_type = NotificationType.ERROR
        
        # Adjust priority based on task priority
        if task.priority == TaskPriority.CRITICAL:
            notification_priority = NotificationPriority.CRITICAL
        elif task.priority == TaskPriority.HIGH and notification_priority != NotificationPriority.CRITICAL:
            notification_priority = NotificationPriority.HIGH
        elif task.priority == TaskPriority.LOW and notification_priority == NotificationPriority.MEDIUM:
            notification_priority = NotificationPriority.LOW
        
        # Include task metadata
        metadata = {
            "task_id": str(task.id),
            "task_title": task.title,
            "old_status": old_status.value if old_status else None,
            "new_status": task.status.value,
            "task_priority": task.priority.value
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
            notification_type = NotificationType.EPIC_STATUS
            priority = NotificationPriority.MEDIUM
        elif progress_pct >= 50:
            message = f"Epic {epic.title} is {progress_pct}% complete. Making good progress."
            notification_type = NotificationType.EPIC_STATUS
            priority = NotificationPriority.MEDIUM
        elif progress_pct >= 25:
            message = f"Epic {epic.title} is {progress_pct}% complete. Moving forward."
            notification_type = NotificationType.EPIC_STATUS
            priority = NotificationPriority.LOW
        else:
            message = f"Epic {epic.title} is {progress_pct}% complete. Just getting started."
            notification_type = NotificationType.EPIC_STATUS
            priority = NotificationPriority.LOW
        
        # Adjust priority based on epic priority
        if epic.priority == TaskPriority.CRITICAL:
            priority = NotificationPriority.CRITICAL
        elif epic.priority == TaskPriority.HIGH and priority != NotificationPriority.CRITICAL:
            priority = NotificationPriority.HIGH
        
        # Include epic metadata
        metadata = {
            "epic_id": str(epic.id),
            "epic_title": epic.title,
            "progress": progress_pct,
            "epic_priority": epic.priority.value
        }
        
        return self.notify(
            message=message,
            notification_type=notification_type,
            priority=priority,
            metadata=metadata
        )
    
    def silence(self, enable_silent_mode: bool) -> None:
        """
        Enable or disable silent mode for notifications.
        
        Args:
            enable_silent_mode: Whether to enable silent mode
        """
        logger.info(f"{'Enabling' if enable_silent_mode else 'Disabling'} silent mode")
        self.config.silent_mode = enable_silent_mode
    
    def set_user_preferences(self, preferences: UserPreferences) -> None:
        """
        Set user preferences for notifications.
        
        Args:
            preferences: User notification preferences
        """
        self.config.user_preferences = preferences
        logger.info(f"Updated user notification preferences")
    
    def update_voice_profile(self, 
                            priority: NotificationPriority, 
                            voice_id: Optional[str] = None, 
                            speed: Optional[float] = None) -> None:
        """
        Update voice profile for a specific priority level.
        
        Args:
            priority: Priority level to update
            voice_id: New voice ID
            speed: New speech speed
        """
        if priority not in self.config.voices:
            self.config.voices[priority] = VoiceProfile()
        
        if voice_id:
            self.config.voices[priority].voice_id = voice_id
        if speed:
            self.config.voices[priority].speed = speed
        
        logger.info(f"Updated voice profile for {priority.value} priority")
    
    def get_history(self, 
                   limit: int = 10, 
                   notification_type: Optional[NotificationType] = None,
                   priority: Optional[NotificationPriority] = None) -> List[Dict[str, Any]]:
        """
        Get recent notification history.
        
        Args:
            limit: Maximum number of items to return
            notification_type: Optional filter by notification type
            priority: Optional filter by priority level
            
        Returns:
            List of notification history items
        """
        if not self.config.history_enabled:
            return []
        
        # Start with all notifications
        filtered = self.history.notifications
        
        # Filter by type if specified
        if notification_type:
            filtered = [n for n in filtered if n.get("type") == notification_type.value]
        
        # Filter by priority if specified
        if priority:
            filtered = [n for n in filtered if n.get("priority") == priority.value]
        
        # Return most recent first, limited by count
        return sorted(filtered, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]
    
    def clear_queue(self) -> int:
        """
        Clear the notification queue.
        
        Returns:
            Number of notifications removed from the queue
        """
        with self.queue_lock:
            queue_size = len(self.notification_queue)
            self.notification_queue = []
            logger.info(f"Cleared notification queue, removed {queue_size} notifications")
            return queue_size
    
    def pause_notifications(self, duration_seconds: float = 300) -> None:
        """
        Pause notifications for a specific duration.
        
        Args:
            duration_seconds: Duration to pause notifications (default: 5 minutes)
        """
        self.config.silent_mode = True
        logger.info(f"Notifications paused for {duration_seconds} seconds")
        
        def resume_notifications():
            time.sleep(duration_seconds)
            self.config.silent_mode = False
            logger.info("Notifications resumed")
        
        # Start a thread to automatically resume notifications
        resume_thread = threading.Thread(target=resume_notifications, daemon=True)
        resume_thread.start()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get the connection status of the TTS service.
        
        Returns:
            Dictionary with connection status details
        """
        # Check connection if it's been a while
        current_time = time.time()
        if current_time - self.last_notification_time > 30.0:
            self.tts_available = self._check_tts_connection()
        
        return {
            "tts_available": self.tts_available,
            "tts_url": self.tts_client.base_url,
            "last_check": self.last_notification_time,
            "notifications_enabled": self.config.enabled,
            "silent_mode": self.config.silent_mode,
            "queue_size": len(self.notification_queue)
        }
    
    def shutdown(self) -> None:
        """
        Clean shutdown of the notification manager.
        """
        logger.info("Shutting down notification manager")
        
        # Stop queue processing
        self.queue_processing_active = False
        if self.queue_processing_thread and self.queue_processing_thread.is_alive():
            try:
                self.queue_processing_thread.join(timeout=5.0)
            except:
                pass
        
        # Save history
        self._save_history()
        
        # Process any remaining high-priority notifications
        with self.queue_lock:
            high_priority_notifications = [
                n for n in self.notification_queue 
                if n.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]
            ]
        
        for notification in high_priority_notifications[:3]:  # Process up to 3 high-priority notifications
            try:
                self._process_single_notification(notification)
            except:
                pass
        
        logger.info("Notification manager shutdown complete")