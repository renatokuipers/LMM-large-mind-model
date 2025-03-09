"""
Experience Logger

This module provides functionality for logging and retrieving experiences
in the LMM system. Experiences include interactions with the Mother,
sensory inputs, emotional states, and developmental milestones.

Experiences are stored with rich metadata to support retrieval by various
dimensions including time, emotional valence, developmental stage, etc.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class ExperienceLogger:
    """
    Records and retrieves experiences for the LMM system.
    
    Experiences are stored in both a SQLite database (for efficient querying)
    and a file-based system (for complex objects and embeddings).
    """
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize the experience logger
        
        Args:
            storage_dir: Base directory for storing experiences
        """
        # Ensure storage directory exists
        self.base_dir = Path(storage_dir)
        self.experiences_dir = self.base_dir / "experiences"
        self.experiences_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.db_path = self.base_dir / "experiences.db"
        self.conn = self._initialize_database()
        
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize the experiences database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create experiences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            type TEXT,
            source TEXT,
            developmental_stage TEXT,
            age REAL,
            emotional_valence TEXT,
            emotional_intensity REAL,
            importance_score REAL,
            tags TEXT,
            metadata TEXT,
            file_path TEXT
        )
        ''')
        
        # Create index on timestamp for efficient retrieval
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp)
        ''')
        
        # Create index on type for efficient filtering
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_type ON experiences(type)
        ''')
        
        # Create index on developmental_stage
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stage ON experiences(developmental_stage)
        ''')
        
        conn.commit()
        return conn
    
    def log_experience(
        self,
        experience_data: Dict[str, Any],
        experience_type: str,
        source: str,
        emotional_valence: str = "neutral",
        emotional_intensity: float = 0.5,
        importance_score: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Log a new experience
        
        Args:
            experience_data: The actual experience data
            experience_type: Type of experience (e.g., 'interaction', 'perception', 'milestone')
            source: Source of the experience (e.g., 'mother', 'self', 'environment')
            emotional_valence: Emotional tone of the experience
            emotional_intensity: Intensity of the emotion (0.0-1.0)
            importance_score: Subjective importance (0.0-1.0)
            tags: List of tags for categorization
            metadata: Additional metadata about the experience
            embedding: Vector embedding of the experience for similarity retrieval
            
        Returns:
            ID of the logged experience
        """
        try:
            # Generate a unique ID based on timestamp
            timestamp = datetime.now().isoformat()
            experience_id = f"exp_{timestamp.replace(':', '-').replace('.', '-')}"
            
            # Prepare tags
            if tags is None:
                tags = []
            tags_str = json.dumps(tags)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata_str = json.dumps(metadata)
            
            # Create file path for the experience data
            file_path = self.experiences_dir / f"{experience_id}.json"
            
            # Save the complete experience to file
            with open(file_path, 'w') as f:
                # Combine all data into a single structure
                full_experience = {
                    "id": experience_id,
                    "timestamp": timestamp,
                    "type": experience_type,
                    "source": source,
                    "emotional_valence": emotional_valence,
                    "emotional_intensity": emotional_intensity,
                    "importance_score": importance_score,
                    "tags": tags,
                    "metadata": metadata,
                    "data": experience_data
                }
                
                # Add embedding if provided
                if embedding is not None:
                    # Convert numpy array to list for JSON serialization
                    full_experience["embedding"] = embedding.tolist()
                    
                json.dump(full_experience, f, indent=2)
            
            # Store summary in the database for efficient querying
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO experiences (
                id, timestamp, type, source, developmental_stage, age,
                emotional_valence, emotional_intensity, importance_score,
                tags, metadata, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience_id,
                timestamp,
                experience_type,
                source,
                metadata.get("developmental_stage", "unknown"),
                metadata.get("age", 0.0),
                emotional_valence,
                emotional_intensity,
                importance_score,
                tags_str,
                metadata_str,
                str(file_path)
            ))
            
            self.conn.commit()
            logger.info(f"Logged experience {experience_id} of type {experience_type}")
            
            return experience_id
            
        except Exception as e:
            logger.error(f"Error logging experience: {e}")
            self.conn.rollback()
            return ""
    
    def get_experience(self, experience_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific experience by ID
        
        Args:
            experience_id: ID of the experience to retrieve
            
        Returns:
            Dictionary containing the experience data
        """
        try:
            # Get file path from database
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM experiences WHERE id = ?", (experience_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Experience {experience_id} not found")
                return {}
                
            file_path = result[0]
            
            # Load experience from file
            with open(file_path, 'r') as f:
                experience = json.load(f)
                
            return experience
            
        except Exception as e:
            logger.error(f"Error retrieving experience {experience_id}: {e}")
            return {}
    
    def query_experiences(
        self,
        experience_type: Optional[str] = None,
        source: Optional[str] = None,
        developmental_stage: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        emotional_valence: Optional[str] = None,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        order_by: str = "timestamp",
        order_direction: str = "DESC"
    ) -> List[Dict[str, Any]]:
        """
        Query experiences based on various criteria
        
        Args:
            experience_type: Filter by experience type
            source: Filter by source
            developmental_stage: Filter by developmental stage
            time_range: Filter by time range (start, end)
            emotional_valence: Filter by emotional valence
            min_importance: Minimum importance score
            tags: Filter by tags (any match)
            limit: Maximum number of results
            order_by: Field to sort by
            order_direction: Sort direction (ASC or DESC)
            
        Returns:
            List of matching experiences
        """
        try:
            query_parts = ["SELECT * FROM experiences WHERE 1=1"]
            params = []
            
            # Add filters
            if experience_type:
                query_parts.append("AND type = ?")
                params.append(experience_type)
                
            if source:
                query_parts.append("AND source = ?")
                params.append(source)
                
            if developmental_stage:
                query_parts.append("AND developmental_stage = ?")
                params.append(developmental_stage)
                
            if time_range:
                start_time, end_time = time_range
                query_parts.append("AND timestamp BETWEEN ? AND ?")
                params.append(start_time.isoformat())
                params.append(end_time.isoformat())
                
            if emotional_valence:
                query_parts.append("AND emotional_valence = ?")
                params.append(emotional_valence)
                
            if min_importance > 0.0:
                query_parts.append("AND importance_score >= ?")
                params.append(min_importance)
                
            if tags:
                # For each tag, check if it exists in the JSON array
                for tag in tags:
                    query_parts.append("AND tags LIKE ?")
                    params.append(f"%{tag}%")  # Simple but imperfect approach
            
            # Add order by clause
            query_parts.append(f"ORDER BY {order_by} {order_direction}")
            
            # Add limit
            query_parts.append("LIMIT ?")
            params.append(limit)
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(" ".join(query_parts), tuple(params))
            results = cursor.fetchall()
            
            # Get column names for dictionary conversion
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert to list of dictionaries
            experiences = []
            for row in results:
                # Create dictionary from row and column names
                exp_dict = dict(zip(column_names, row))
                
                # Load full experience data from file
                try:
                    with open(exp_dict["file_path"], 'r') as f:
                        full_experience = json.load(f)
                        experiences.append(full_experience)
                except FileNotFoundError:
                    # If file is missing, just return the database record
                    experiences.append(exp_dict)
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error querying experiences: {e}")
            return []
    
    def update_experience_metadata(
        self,
        experience_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for an experience
        
        Args:
            experience_id: ID of the experience to update
            metadata_updates: New metadata values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current experience
            experience = self.get_experience(experience_id)
            if not experience:
                return False
                
            # Update metadata
            if "metadata" not in experience:
                experience["metadata"] = {}
                
            experience["metadata"].update(metadata_updates)
            
            # Save updated experience back to file
            with open(experience.get("file_path", ""), 'w') as f:
                json.dump(experience, f, indent=2)
                
            # Update database record
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE experiences SET metadata = ? WHERE id = ?",
                (json.dumps(experience["metadata"]), experience_id)
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating experience metadata: {e}")
            self.conn.rollback()
            return False
    
    def get_experience_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        group_by_day: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get a timeline of experiences
        
        Args:
            start_time: Start time for the timeline
            end_time: End time for the timeline
            limit: Maximum number of experiences to retrieve
            group_by_day: Group experiences by day
            
        Returns:
            List of experiences or grouped experiences
        """
        try:
            # Set default time range if not provided
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
                
            # Base query
            query = """
            SELECT * FROM experiences 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(query, (start_time.isoformat(), end_time.isoformat(), limit))
            results = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert to list of dictionaries
            experiences = [dict(zip(column_names, row)) for row in results]
            
            # Group by day if requested
            if group_by_day:
                grouped = {}
                for exp in experiences:
                    timestamp = datetime.fromisoformat(exp["timestamp"])
                    day_key = timestamp.date().isoformat()
                    
                    if day_key not in grouped:
                        grouped[day_key] = []
                        
                    grouped[day_key].append(exp)
                    
                # Convert to list sorted by day
                return [{"date": day, "experiences": exps} for day, exps in sorted(grouped.items(), reverse=True)]
                
            return experiences
            
        except Exception as e:
            logger.error(f"Error retrieving experience timeline: {e}")
            return []
    
    def delete_experience(self, experience_id: str) -> bool:
        """
        Delete an experience
        
        Args:
            experience_id: ID of the experience to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get file path
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM experiences WHERE id = ?", (experience_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Experience {experience_id} not found for deletion")
                return False
                
            file_path = result[0]
            
            # Delete from database
            cursor.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
            self.conn.commit()
            
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            logger.info(f"Deleted experience {experience_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting experience {experience_id}: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed on deletion"""
        self.close() 
