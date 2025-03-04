# session_manager.py - Manages training interaction sessions
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging
import json
import os
from pathlib import Path
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator

from utils.logging_utils import log_metrics

logger = logging.getLogger("SessionManager")

class InteractionRecord(BaseModel):
    """Record of a single interaction"""
    timestamp: datetime = Field(default_factory=datetime.now)
    child_message: str
    mother_message: str
    child_emotion: str
    mother_emotion: str
    vocabulary_size: int = Field(0, ge=0)
    age_days: float = Field(0.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmotionalProgress(BaseModel):
    """Emotional progress during a session"""
    initial_emotions: Dict[str, float] = Field(default_factory=dict)
    final_emotions: Dict[str, float] = Field(default_factory=dict)
    emotional_growth: Dict[str, float] = Field(default_factory=dict)
    emotional_stability: float = Field(0.0, ge=0.0, le=1.0)
    dominant_emotions: List[Tuple[str, float]] = Field(default_factory=list)

class VocabularyProgress(BaseModel):
    """Vocabulary progress during a session"""
    initial_size: int = Field(0, ge=0)
    final_size: int = Field(0, ge=0)
    new_words: List[str] = Field(default_factory=list)
    learning_rate: float = Field(0.0, ge=0.0)
    category_growth: Dict[str, int] = Field(default_factory=dict)

class TrainingSession(BaseModel):
    """Information about a training session"""
    session_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    interactions: List[InteractionRecord] = Field(default_factory=list)
    emotional_progress: Optional[EmotionalProgress] = None
    vocabulary_progress: Optional[VocabularyProgress] = None
    notes: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def calculate_duration(self) -> 'TrainingSession':
        """Calculate duration if end_time is set"""
        if self.end_time and not self.duration_minutes:
            self.duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        return self

class SessionManager:
    """Manages training interaction sessions"""
    
    def __init__(self, save_dir: Path = Path("./data/training")):
        """Initialize session manager
        
        Args:
            save_dir: Directory for saving session data
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Active and historical sessions
        self.current_session: Optional[TrainingSession] = None
        self.sessions: Dict[str, TrainingSession] = {}
        
        # Initialize stats
        self.total_interactions = 0
        self.total_session_time = 0.0  # minutes
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"Session manager initialized with {len(self.sessions)} historical sessions")
    
    def _load_sessions(self) -> None:
        """Load existing sessions from files"""
        session_files = list(self.save_dir.glob("session_*.json"))
        
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                
                # Convert timestamps to datetime objects
                if "start_time" in session_data:
                    session_data["start_time"] = datetime.fromisoformat(session_data["start_time"])
                if "end_time" in session_data and session_data["end_time"]:
                    session_data["end_time"] = datetime.fromisoformat(session_data["end_time"])
                
                # Convert timestamps in interactions
                if "interactions" in session_data:
                    for interaction in session_data["interactions"]:
                        if "timestamp" in interaction:
                            interaction["timestamp"] = datetime.fromisoformat(interaction["timestamp"])
                
                # Create session object
                session = TrainingSession(**session_data)
                self.sessions[session.session_id] = session
                
                # Update stats
                self.total_interactions += len(session.interactions)
                self.total_session_time += session.duration_minutes or 0
                
            except Exception as e:
                logger.error(f"Error loading session from {file_path}: {str(e)}")
    
    def save_state(self, custom_path: Optional[Path] = None) -> None:
        """Save current state of session manager
        
        Args:
            custom_path: Custom path for saving state
        """
        if custom_path:
            state_path = custom_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_path = self.save_dir / f"session_manager_{timestamp}.json"
        
        try:
            # Prepare state data
            state = {
                "total_interactions": self.total_interactions,
                "total_session_time": self.total_session_time,
                "current_session_id": self.current_session.session_id if self.current_session else None,
                "session_count": len(self.sessions),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Session manager state saved to {state_path}")
            
        except Exception as e:
            logger.error(f"Error saving session manager state: {str(e)}")
    
    def load_state(self, state_path: Path) -> bool:
        """Load session manager state
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(state_path):
            logger.error(f"State file not found: {state_path}")
            return False
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Update stats
            self.total_interactions = state.get("total_interactions", 0)
            self.total_session_time = state.get("total_session_time", 0.0)
            
            # Load current session if one was active
            current_session_id = state.get("current_session_id")
            if current_session_id and current_session_id in self.sessions:
                self.current_session = self.sessions[current_session_id]
                logger.info(f"Restored current session: {current_session_id}")
            
            logger.info(f"Session manager state loaded from {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session manager state: {str(e)}")
            return False
    
    def start_session(self) -> str:
        """Start a new training session
        
        Returns:
            Session ID
        """
        # Check if there's already an active session
        if self.current_session and not self.current_session.end_time:
            logger.warning(f"Ending current session before starting new one")
            self.end_session(self.current_session.session_id)
        
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Create new session
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # Add to sessions dictionary
        self.sessions[session_id] = self.current_session
        
        logger.info(f"Started new training session: {session_id}")
        return session_id
    
    def end_session(self, session_id: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """End a training session
        
        Args:
            session_id: Session ID to end
            metrics: Additional metrics to record
        """
        if session_id not in self.sessions:
            logger.error(f"Session not found: {session_id}")
            return
        
        session = self.sessions[session_id]
        
        # Set end time if not already set
        if not session.end_time:
            session.end_time = datetime.now()
            # Calculate duration
            session.duration_minutes = (session.end_time - session.start_time).total_seconds() / 60
            
            # Update total session time
            self.total_session_time += session.duration_minutes
        
        # Calculate progress metrics
        if session.interactions:
            session.emotional_progress = self._calculate_emotional_progress(session)
            session.vocabulary_progress = self._calculate_vocabulary_progress(session)
        
        # Add custom metrics if provided
        if metrics:
            session.metrics.update(metrics)
        
        # Save session to file
        self._save_session(session)
        
        # Clear current session if this was it
        if self.current_session and self.current_session.session_id == session_id:
            self.current_session = None
        
        logger.info(f"Ended training session: {session_id}, duration: {session.duration_minutes:.2f} minutes")
        
        # Log metrics
        log_metrics({
            "session_id": session_id,
            "duration_minutes": session.duration_minutes,
            "interaction_count": len(session.interactions),
            "vocabulary_growth": session.vocabulary_progress.final_size - session.vocabulary_progress.initial_size if session.vocabulary_progress else 0
        }, "training_session")
    
    def _save_session(self, session: TrainingSession) -> None:
        """Save session to file
        
        Args:
            session: Session to save
        """
        file_path = self.save_dir / f"session_{session.session_id}.json"
        
        try:
            # Serialize session to JSON
            with open(file_path, 'w') as f:
                json.dump(session.model_dump(), f, indent=2, default=str)
                
            logger.info(f"Saved session to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving session to {file_path}: {str(e)}")
    
    def record_interaction(
        self,
        session_id: str,
        child_message: str,
        mother_message: str,
        child_emotion: str,
        mother_emotion: str,
        vocabulary_size: int,
        age_days: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an interaction in a session
        
        Args:
            session_id: Session ID
            child_message: Message from the child
            mother_message: Message from the mother
            child_emotion: Child's emotion
            mother_emotion: Mother's emotion
            vocabulary_size: Current vocabulary size
            age_days: Current age in days
            metadata: Additional metadata
        """
        if session_id not in self.sessions:
            logger.error(f"Session not found: {session_id}")
            return
        
        session = self.sessions[session_id]
        
        # Create interaction record
        interaction = InteractionRecord(
            timestamp=datetime.now(),
            child_message=child_message,
            mother_message=mother_message,
            child_emotion=child_emotion,
            mother_emotion=mother_emotion,
            vocabulary_size=vocabulary_size,
            age_days=age_days,
            metadata=metadata or {}
        )
        
        # Add to session
        session.interactions.append(interaction)
        
        # Update stats
        self.total_interactions += 1
        
        logger.debug(f"Recorded interaction in session {session_id}")
        
        # Save session periodically
        if len(session.interactions) % 50 == 0:
            self._save_session(session)
    
    def _calculate_emotional_progress(self, session: TrainingSession) -> EmotionalProgress:
        """Calculate emotional progress during a session
        
        Args:
            session: Training session
            
        Returns:
            Emotional progress data
        """
        # Need at least two interactions to calculate progress
        if len(session.interactions) < 2:
            return EmotionalProgress()
        
        # Extract emotions
        emotions_by_interaction = []
        for interaction in session.interactions:
            # Convert emotion string to a simple dictionary
            if interaction.child_emotion:
                emotion_dict = {interaction.child_emotion: 1.0}
                emotions_by_interaction.append(emotion_dict)
        
        if not emotions_by_interaction:
            return EmotionalProgress()
        
        # Calculate initial and final emotions (using first/last 3 interactions)
        initial_range = min(3, len(emotions_by_interaction))
        final_range = min(3, len(emotions_by_interaction))
        
        initial_emotions = {}
        for i in range(initial_range):
            for emotion, value in emotions_by_interaction[i].items():
                if emotion not in initial_emotions:
                    initial_emotions[emotion] = 0
                initial_emotions[emotion] += value / initial_range
        
        final_emotions = {}
        for i in range(-final_range, 0):
            for emotion, value in emotions_by_interaction[i].items():
                if emotion not in final_emotions:
                    final_emotions[emotion] = 0
                final_emotions[emotion] += value / final_range
        
        # Calculate emotional growth
        emotional_growth = {}
        for emotion in set(initial_emotions.keys()) | set(final_emotions.keys()):
            initial = initial_emotions.get(emotion, 0)
            final = final_emotions.get(emotion, 0)
            emotional_growth[emotion] = final - initial
        
        # Calculate emotional stability
        emotions_counts = {}
        for emotion_dict in emotions_by_interaction:
            for emotion in emotion_dict:
                if emotion not in emotions_counts:
                    emotions_counts[emotion] = 0
                emotions_counts[emotion] += 1
        
        # Higher stability when fewer emotions or more consistent emotions
        if emotions_counts and len(session.interactions) > 0:
            max_count = max(emotions_counts.values())
            stability = max_count / len(session.interactions)
        else:
            stability = 0.5  # Default value
        
        # Find dominant emotions
        dominant_emotions = sorted(
            [(emotion, count / len(session.interactions)) 
             for emotion, count in emotions_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3
        
        return EmotionalProgress(
            initial_emotions=initial_emotions,
            final_emotions=final_emotions,
            emotional_growth=emotional_growth,
            emotional_stability=stability,
            dominant_emotions=dominant_emotions
        )
    
    def _calculate_vocabulary_progress(self, session: TrainingSession) -> VocabularyProgress:
        """Calculate vocabulary progress during a session
        
        Args:
            session: Training session
            
        Returns:
            Vocabulary progress data
        """
        # Need at least two interactions to calculate progress
        if len(session.interactions) < 2:
            return VocabularyProgress()
        
        # Get initial and final vocabulary sizes
        initial_size = session.interactions[0].vocabulary_size
        final_size = session.interactions[-1].vocabulary_size
        
        # Calculate learning rate
        session_minutes = (session.interactions[-1].timestamp - session.interactions[0].timestamp).total_seconds() / 60
        if session_minutes > 0:
            learning_rate = (final_size - initial_size) / session_minutes
        else:
            learning_rate = 0
        
        # We don't have access to the actual vocabulary, so we can't determine new words
        # This would need to be provided via metadata
        
        return VocabularyProgress(
            initial_size=initial_size,
            final_size=final_size,
            learning_rate=learning_rate
        )
    
    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get a session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID
        
        Returns:
            Current session ID or None if no active session
        """
        if self.current_session:
            return self.current_session.session_id
        return None
    
    def get_recent_sessions(self, count: int = 5) -> List[TrainingSession]:
        """Get the most recent sessions
        
        Args:
            count: Number of sessions to return
            
        Returns:
            List of recent sessions
        """
        # Sort sessions by start time
        sorted_sessions = sorted(
            self.sessions.values(), 
            key=lambda s: s.start_time, 
            reverse=True
        )
        
        return sorted_sessions[:count]
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary if found, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Prepare summary
        summary = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "duration_minutes": session.duration_minutes,
            "interaction_count": len(session.interactions),
            "emotional_progress": session.emotional_progress.model_dump() if session.emotional_progress else None,
            "vocabulary_progress": session.vocabulary_progress.model_dump() if session.vocabulary_progress else None,
            "metrics": session.metrics
        }
        
        return summary
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all sessions
        
        Returns:
            Global statistics
        """
        # Calculate total training time
        total_hours = self.total_session_time / 60
        
        # Count completed sessions
        completed_sessions = sum(1 for s in self.sessions.values() if s.end_time is not None)
        
        # Calculate average session length
        if completed_sessions > 0:
            avg_session_minutes = self.total_session_time / completed_sessions
        else:
            avg_session_minutes = 0
        
        # Calculate average interactions per session
        if completed_sessions > 0:
            avg_interactions = self.total_interactions / completed_sessions
        else:
            avg_interactions = 0
        
        # Calculate vocabulary growth rate across all sessions
        total_vocab_growth = 0
        sessions_with_vocab_data = 0
        
        for session in self.sessions.values():
            if session.vocabulary_progress:
                vocab_growth = session.vocabulary_progress.final_size - session.vocabulary_progress.initial_size
                if vocab_growth > 0:
                    total_vocab_growth += vocab_growth
                    sessions_with_vocab_data += 1
        
        if sessions_with_vocab_data > 0:
            avg_vocab_growth = total_vocab_growth / sessions_with_vocab_data
        else:
            avg_vocab_growth = 0
        
        return {
            "total_sessions": len(self.sessions),
            "completed_sessions": completed_sessions,
            "total_interactions": self.total_interactions,
            "total_training_hours": total_hours,
            "avg_session_minutes": avg_session_minutes,
            "avg_interactions_per_session": avg_interactions,
            "avg_vocabulary_growth_per_session": avg_vocab_growth,
            "current_session_active": self.current_session is not None
        }
    
    def export_interactions_to_csv(self, output_file: Optional[Path] = None) -> Path:
        """Export all interactions to CSV
        
        Args:
            output_file: Output file path (default: interactions.csv in save directory)
            
        Returns:
            Path to output file
        """
        import csv
        
        if output_file is None:
            output_file = self.save_dir / "interactions.csv"
        
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    "session_id", "timestamp", "child_message", "mother_message",
                    "child_emotion", "mother_emotion", "vocabulary_size", "age_days"
                ])
                
                # Write interactions
                for session_id, session in self.sessions.items():
                    for interaction in session.interactions:
                        writer.writerow([
                            session_id,
                            interaction.timestamp.isoformat(),
                            interaction.child_message,
                            interaction.mother_message,
                            interaction.child_emotion,
                            interaction.mother_emotion,
                            interaction.vocabulary_size,
                            interaction.age_days
                        ])
            
            logger.info(f"Exported interactions to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting interactions: {str(e)}")
            return output_file