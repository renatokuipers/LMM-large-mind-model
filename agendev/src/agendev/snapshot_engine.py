# snapshot_engine.py
"""Binary differential storage for local version control."""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Tuple, Iterator
from datetime import datetime
from pathlib import Path
import os
import difflib
import hashlib
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator

from .utils.fs_utils import (
    resolve_path, save_snapshot, list_snapshots, get_latest_snapshot,
    load_json, save_json, safe_save_json, content_hash
)

class SnapshotMetadata(BaseModel):
    """Metadata for a code snapshot."""
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    file_path: str
    timestamp: datetime = Field(default_factory=datetime.now)
    hash: str
    parent_snapshot_id: Optional[str] = None
    message: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: str = "agendev"
    
    @model_validator(mode='after')
    def validate_snapshot(self) -> 'SnapshotMetadata':
        """Ensure snapshot has required fields."""
        if not self.file_path or not self.hash:
            raise ValueError("Snapshot must have file_path and hash")
        return self

class SnapshotDiff(BaseModel):
    """Represents differences between snapshots."""
    source_id: str
    target_id: str
    added_lines: int = 0
    removed_lines: int = 0
    changed_lines: int = 0
    diff_content: str
    
    def summary(self) -> str:
        """Get a human-readable summary of the changes."""
        return f"+{self.added_lines} -{self.removed_lines} ~{self.changed_lines}"

class Branch(BaseModel):
    """Represents a development branch."""
    name: str
    head_snapshot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    snapshots: List[str] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_branch(self) -> 'Branch':
        """Validate branch data and set defaults."""
        # Ensure created_at is a valid datetime
        if not self.created_at or isinstance(self.created_at, str) and not self.created_at:
            self.created_at = datetime.now()
        return self

class SnapshotEngine:
    """Manages code snapshots and version history."""
    
    def __init__(self, workspace_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the snapshot engine.
        
        Args:
            workspace_dir: Optional workspace directory override
        """
        self.workspace_dir = resolve_path(workspace_dir or "")
        self.snapshots_dir = resolve_path("artifacts/snapshots", create_parents=True)
        self.metadata_dir = self.snapshots_dir / "metadata"
        self.branches_file = self.snapshots_dir / "branches.json"
        
        # Create directories if they don't exist
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize branches
        self.branches: Dict[str, Branch] = {}
        self._load_branches()
        
        # Ensure main branch exists
        if "main" not in self.branches:
            self.create_branch("main")
    
    def _load_branches(self) -> None:
        """Load branch information from disk."""
        if not self.branches_file.exists():
            return
            
        branches_data = load_json(self.branches_file)
        for branch_name, branch_data in branches_data.get("branches", {}).items():
            # Fix empty created_at
            if "created_at" in branch_data and (not branch_data["created_at"] or branch_data["created_at"] == ""):
                branch_data["created_at"] = datetime.now().isoformat()
            try:
                self.branches[branch_name] = Branch.model_validate(branch_data)
            except Exception as e:
                # If validation fails, create a new branch with default values
                self.branches[branch_name] = Branch(
                    name=branch_name,
                    head_snapshot_id=branch_data.get("head_snapshot_id"),
                    snapshots=branch_data.get("snapshots", [])
                )
    
    def _save_branches(self) -> None:
        """Save branch information to disk."""
        branches_data = {
            "branches": {
                name: branch.model_dump() for name, branch in self.branches.items()
            }
        }
        safe_save_json(branches_data, self.branches_file)
    
    def create_snapshot(
        self,
        file_path: Union[str, Path],
        content: str,
        branch: str = "main",
        message: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> SnapshotMetadata:
        """
        Create a snapshot of a file.
        
        Args:
            file_path: Path to the file to snapshot
            content: Content of the file
            branch: Branch to create the snapshot on
            message: Optional snapshot message
            tags: Optional tags for the snapshot
            
        Returns:
            Metadata for the created snapshot
        """
        # Resolve file path
        resolved_path = Path(file_path)
        relative_path = resolved_path.name if resolved_path.is_absolute() else str(resolved_path)
        
        # Ensure branch exists
        if branch not in self.branches:
            self.create_branch(branch)
        
        # Get previous snapshot if any
        parent_id = None
        if self.branches[branch].head_snapshot_id:
            parent_id = self.branches[branch].head_snapshot_id
        
        # Create snapshot
        hash_value, snapshot_path = save_snapshot(
            content=content,
            file_path=file_path,
            metadata={
                "branch": branch,
                "message": message,
                "tags": tags or []
            }
        )
        
        # Create metadata
        metadata = SnapshotMetadata(
            file_path=relative_path,
            hash=hash_value,
            parent_snapshot_id=parent_id,
            message=message,
            tags=tags or []
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{metadata.snapshot_id}.json"
        safe_save_json(metadata.model_dump(), metadata_path)
        
        # Update branch
        branch_obj = self.branches[branch]
        branch_obj.head_snapshot_id = metadata.snapshot_id
        branch_obj.snapshots.append(metadata.snapshot_id)
        self._save_branches()
        
        return metadata
    
    def get_snapshots(self, file_path: Union[str, Path], branch: Optional[str] = None) -> List[SnapshotMetadata]:
        """
        Get snapshots for a file.
        
        Args:
            file_path: Path to the file
            branch: Optional branch to filter by
            
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        file_path_str = str(Path(file_path).name if Path(file_path).is_absolute() else file_path)
        
        # List all snapshot metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                data = load_json(metadata_file)
                metadata = SnapshotMetadata.model_validate(data)
                
                # Filter by file path
                if metadata.file_path == file_path_str:
                    # Filter by branch if specified
                    if branch is None or metadata.snapshot_id in self.branches.get(branch, Branch(name="")).snapshots:
                        snapshots.append(metadata)
            except Exception as e:
                print(f"Error loading snapshot metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
    
    def get_snapshot_content(self, snapshot_id: str) -> Optional[str]:
        """
        Get the content of a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Content of the snapshot or None if not found
        """
        # Find metadata for the snapshot
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        if not metadata_path.exists():
            return None
            
        metadata_data = load_json(metadata_path)
        metadata = SnapshotMetadata.model_validate(metadata_data)
        
        # Try to find the snapshot file
        snapshot_glob = list(self.snapshots_dir.glob(f"**/*_{metadata.hash[:8]}*"))
        if not snapshot_glob:
            return None
            
        snapshot_path = snapshot_glob[0]
        
        try:
            with open(snapshot_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading snapshot {snapshot_id}: {e}")
            return None
    
    def get_all_snapshots(self, branch: Optional[str] = None) -> List[SnapshotMetadata]:
        """
        Get all snapshots across all files.
        
        Args:
            branch: Optional branch to filter by
            
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        
        # List all snapshot metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                data = load_json(metadata_file)
                metadata = SnapshotMetadata.model_validate(data)
                
                # Filter by branch if specified
                if branch is None or metadata.snapshot_id in self.branches.get(branch, Branch(name="")).snapshots:
                    snapshots.append(metadata)
            except Exception as e:
                print(f"Error loading snapshot metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
    
    def compare_snapshots(self, source_id: str, target_id: str) -> Optional[SnapshotDiff]:
        """
        Compare two snapshots.
        
        Args:
            source_id: ID of the source snapshot
            target_id: ID of the target snapshot
            
        Returns:
            Snapshot diff or None if either snapshot not found
        """
        source_content = self.get_snapshot_content(source_id)
        target_content = self.get_snapshot_content(target_id)
        
        if source_content is None or target_content is None:
            return None
            
        # Generate diff
        source_lines = source_content.splitlines()
        target_lines = target_content.splitlines()
        
        diff = list(difflib.unified_diff(
            source_lines,
            target_lines,
            fromfile=f"snapshot_{source_id}",
            tofile=f"snapshot_{target_id}",
            lineterm=''
        ))
        
        # Count changes
        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        changed_lines = min(added_lines, removed_lines)  # Estimate
        added_lines -= changed_lines
        removed_lines -= changed_lines
        
        return SnapshotDiff(
            source_id=source_id,
            target_id=target_id,
            added_lines=added_lines,
            removed_lines=removed_lines,
            changed_lines=changed_lines,
            diff_content='\n'.join(diff)
        )
    
    def create_branch(self, name: str, base_branch: str = "main") -> Branch:
        """
        Create a new branch.
        
        Args:
            name: Name of the branch
            base_branch: Branch to base the new branch on
            
        Returns:
            The created branch
        """
        if name in self.branches:
            return self.branches[name]
            
        head_id = None
        if base_branch in self.branches and self.branches[base_branch].head_snapshot_id:
            head_id = self.branches[base_branch].head_snapshot_id
            
        branch = Branch(name=name, head_snapshot_id=head_id)
        
        if head_id:
            branch.snapshots.append(head_id)
            
        self.branches[name] = branch
        self._save_branches()
        
        return branch
    
    def switch_branch(self, branch: str) -> bool:
        """
        Switch to a branch.
        
        Args:
            branch: Name of the branch to switch to
            
        Returns:
            Whether the switch was successful
        """
        if branch not in self.branches:
            return False
            
        # Nothing special to do here as branches are just pointers
        return True
    
    def get_branch_history(self, branch: str) -> List[SnapshotMetadata]:
        """
        Get the history of a branch.
        
        Args:
            branch: Name of the branch
            
        Returns:
            List of snapshot metadata in chronological order
        """
        if branch not in self.branches:
            return []
            
        snapshots = []
        for snapshot_id in self.branches[branch].snapshots:
            metadata_path = self.metadata_dir / f"{snapshot_id}.json"
            if metadata_path.exists():
                data = load_json(metadata_path)
                snapshots.append(SnapshotMetadata.model_validate(data))
        
        # Sort by timestamp
        return sorted(snapshots, key=lambda x: x.timestamp)
    
    def restore_snapshot(self, snapshot_id: str, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Restore a snapshot to a file.
        
        Args:
            snapshot_id: ID of the snapshot to restore
            output_path: Optional path to restore to
            
        Returns:
            Path to the restored file or None if snapshot not found
        """
        content = self.get_snapshot_content(snapshot_id)
        if content is None:
            return None
            
        # Find metadata for the snapshot
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        if not metadata_path.exists():
            return None
            
        metadata_data = load_json(metadata_path)
        metadata = SnapshotMetadata.model_validate(metadata_data)
        
        # Determine output path
        if output_path is None:
            output_path = resolve_path(metadata.file_path)
            
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Write content
        with open(output_path, 'w') as f:
            f.write(content)
            
        return str(output_path)