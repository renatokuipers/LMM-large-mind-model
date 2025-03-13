# snapshot_engine.py
"""Binary differential storage for local version control."""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any, Tuple, Iterator, Set
from datetime import datetime
from pathlib import Path
import os
import difflib
import hashlib
import bz2
import binascii
from uuid import uuid4
from pydantic import BaseModel, Field, model_validator
import logging

from .utils.fs_utils import (
    resolve_path, save_snapshot, list_snapshots, get_latest_snapshot,
    load_json, save_json, safe_save_json, content_hash
)

logger = logging.getLogger(__name__)

class DiffPatch(BaseModel):
    """Binary differential patch between two snapshots."""
    operation: str  # 'add', 'delete', 'replace'
    position: int
    content: str = ""
    
    def apply(self, source: str) -> str:
        """Apply this patch to a source string."""
        if self.operation == 'add':
            return source[:self.position] + self.content + source[self.position:]
        elif self.operation == 'delete':
            return source[:self.position] + source[self.position + len(self.content):]
        elif self.operation == 'replace':
            return source[:self.position] + self.content + source[self.position + len(self.content):]
        return source

class SnapshotMetadata(BaseModel):
    """Metadata for a code snapshot."""
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    file_path: str
    timestamp: datetime = Field(default_factory=datetime.now)
    hash: str
    size: int = 0
    parent_snapshot_id: Optional[str] = None
    message: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: str = "agendev"
    status: str = "success"  # success, failed, reverted
    execution_time: Optional[float] = None
    language: Optional[str] = None
    compressed: bool = False
    
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
    patches: Optional[List[DiffPatch]] = None
    
    def summary(self) -> str:
        """Get a human-readable summary of the changes."""
        return f"+{self.added_lines} -{self.removed_lines} ~{self.changed_lines}"
    
    def has_conflicts(self) -> bool:
        """Check if the diff has potential conflicts."""
        # Simple implementation - if there are both added and removed lines, 
        # there's a higher chance of conflicts
        return self.added_lines > 0 and self.removed_lines > 0

class Branch(BaseModel):
    """Represents a development branch."""
    name: str
    head_snapshot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    snapshots: List[str] = Field(default_factory=list)
    protected: bool = False  # Prevent accidental deletion
    description: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_branch(self) -> 'Branch':
        """Validate branch data and set defaults."""
        # Ensure created_at is a valid datetime
        if not self.created_at or isinstance(self.created_at, str) and not self.created_at:
            self.created_at = datetime.now()
        return self

class SnapshotError(Exception):
    """Error related to snapshot operations."""
    pass

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
        self.diffs_dir = self.snapshots_dir / "diffs"
        
        # Create directories if they don't exist
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.diffs_dir, exist_ok=True)
        
        # Initialize branches
        self.branches: Dict[str, Branch] = {}
        self._load_branches()
        
        # Ensure main branch exists
        if "main" not in self.branches:
            self.create_branch("main")
        
        # Cache for recently accessed snapshots to improve performance
        self._snapshot_cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, SnapshotMetadata] = {}
        self.cache_size = 20  # Number of snapshots to keep in memory
    
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
                logger.warning(f"Error validating branch {branch_name}: {e}")
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
        success = safe_save_json(branches_data, self.branches_file)
        if not success:
            logger.error("Failed to save branch information")
    
    def _clear_cache(self) -> None:
        """Clear the snapshot cache."""
        self._snapshot_cache = {}
        self._metadata_cache = {}
    
    def _add_to_cache(self, snapshot_id: str, content: str, metadata: Optional[SnapshotMetadata] = None) -> None:
        """Add a snapshot to the cache."""
        # Maintain cache size by removing oldest items
        if len(self._snapshot_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._snapshot_cache))
            del self._snapshot_cache[oldest_key]
            if oldest_key in self._metadata_cache:
                del self._metadata_cache[oldest_key]
        
        self._snapshot_cache[snapshot_id] = content
        if metadata:
            self._metadata_cache[snapshot_id] = metadata
    
    def _compress_content(self, content: str) -> bytes:
        """Compress content with BZ2."""
        return bz2.compress(content.encode('utf-8'))
    
    def _decompress_content(self, data: bytes) -> str:
        """Decompress BZ2 data to string."""
        return bz2.decompress(data).decode('utf-8')
    
    def _compute_binary_diff(self, source: str, target: str) -> List[DiffPatch]:
        """
        Compute binary differences between source and target strings.
        
        Uses a simplified algorithm that identifies changed blocks.
        """
        patches = []
        matcher = difflib.SequenceMatcher(None, source, target)
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                continue
            elif op == 'insert':
                patches.append(DiffPatch(
                    operation='add',
                    position=i1,
                    content=target[j1:j2]
                ))
            elif op == 'delete':
                patches.append(DiffPatch(
                    operation='delete',
                    position=i1,
                    content=source[i1:i2]
                ))
            elif op == 'replace':
                patches.append(DiffPatch(
                    operation='replace',
                    position=i1,
                    content=target[j1:j2]
                ))
        
        return patches
    
    def _apply_patches(self, source: str, patches: List[DiffPatch]) -> str:
        """Apply a sequence of patches to a source string."""
        result = source
        for patch in patches:
            result = patch.apply(result)
        return result
    
    def create_snapshot(
        self,
        file_path: Union[str, Path],
        content: str,
        branch: str = "main",
        message: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: str = "success",
        language: Optional[str] = None,
        execution_time: Optional[float] = None,
        compress: bool = False
    ) -> SnapshotMetadata:
        """
        Create a snapshot of a file.
        
        Args:
            file_path: Path to the file to snapshot
            content: Content of the file
            branch: Branch to create the snapshot on
            message: Optional snapshot message
            tags: Optional tags for the snapshot
            status: Status of the snapshot ("success", "failed", "reverted")
            language: Programming language of the content
            execution_time: Time taken to execute the operation
            compress: Whether to compress the content
            
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
        
        # Create a hash of the content
        hash_value, snapshot_path = save_snapshot(
            content=content,
            file_path=file_path,
            metadata={
                "branch": branch,
                "message": message,
                "tags": tags or [],
                "status": status,
                "language": language,
                "execution_time": execution_time,
                "compressed": compress
            }
        )
        
        # Create metadata
        metadata = SnapshotMetadata(
            file_path=relative_path,
            hash=hash_value,
            size=len(content),
            parent_snapshot_id=parent_id,
            message=message,
            tags=tags or [],
            status=status,
            language=language,
            execution_time=execution_time,
            compressed=compress
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{metadata.snapshot_id}.json"
        success = safe_save_json(metadata.model_dump(), metadata_path)
        if not success:
            raise SnapshotError(f"Failed to save metadata for snapshot {metadata.snapshot_id}")
        
        # Update branch
        branch_obj = self.branches[branch]
        branch_obj.head_snapshot_id = metadata.snapshot_id
        branch_obj.snapshots.append(metadata.snapshot_id)
        self._save_branches()
        
        # If there's a parent, save the binary diff
        if parent_id:
            parent_content = self.get_snapshot_content(parent_id)
            if parent_content:
                # Compute binary diff
                diff_patches = self._compute_binary_diff(parent_content, content)
                
                # Save diff to file
                diff_file = self.diffs_dir / f"{parent_id}_{metadata.snapshot_id}.json"
                diff_data = {
                    "source_id": parent_id,
                    "target_id": metadata.snapshot_id,
                    "patches": [patch.model_dump() for patch in diff_patches]
                }
                safe_save_json(diff_data, diff_file)
        
        # Cache the new snapshot
        self._add_to_cache(metadata.snapshot_id, content, metadata)
        
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
                logger.error(f"Error loading snapshot metadata {metadata_file}: {e}")
        
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
        # Check cache first
        if snapshot_id in self._snapshot_cache:
            return self._snapshot_cache[snapshot_id]
        
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
            with open(snapshot_path, 'rb' if metadata.compressed else 'r') as f:
                if metadata.compressed:
                    content = self._decompress_content(f.read())
                else:
                    content = f.read()
                
                # Cache the result
                self._add_to_cache(snapshot_id, content, metadata)
                return content
        except Exception as e:
            logger.error(f"Error reading snapshot {snapshot_id}: {e}")
            return None
    
    def get_snapshot_metadata(self, snapshot_id: str) -> Optional[SnapshotMetadata]:
        """
        Get metadata for a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Snapshot metadata or None if not found
        """
        # Check cache first
        if snapshot_id in self._metadata_cache:
            return self._metadata_cache[snapshot_id]
        
        # Find metadata for the snapshot
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        if not metadata_path.exists():
            return None
            
        try:
            data = load_json(metadata_path)
            metadata = SnapshotMetadata.model_validate(data)
            
            # Cache the result
            self._metadata_cache[snapshot_id] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata for snapshot {snapshot_id}: {e}")
            return None
    
    def get_all_snapshots(self, branch: Optional[str] = None, status: Optional[str] = None) -> List[SnapshotMetadata]:
        """
        Get all snapshots across all files.
        
        Args:
            branch: Optional branch to filter by
            status: Optional status to filter by
            
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
                branch_match = branch is None or metadata.snapshot_id in self.branches.get(branch, Branch(name="")).snapshots
                
                # Filter by status if specified
                status_match = status is None or metadata.status == status
                
                if branch_match and status_match:
                    snapshots.append(metadata)
            except Exception as e:
                logger.error(f"Error loading snapshot metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
    
    def compare_snapshots(self, source_id: str, target_id: str, compute_binary_diff: bool = True) -> Optional[SnapshotDiff]:
        """
        Compare two snapshots.
        
        Args:
            source_id: ID of the source snapshot
            target_id: ID of the target snapshot
            compute_binary_diff: Whether to compute binary diff patches
            
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
        
        patches = None
        if compute_binary_diff:
            # Check if we have a cached diff
            diff_file = self.diffs_dir / f"{source_id}_{target_id}.json"
            if diff_file.exists():
                diff_data = load_json(diff_file)
                patches = [DiffPatch.model_validate(patch) for patch in diff_data.get("patches", [])]
            else:
                # Compute binary diff
                patches = self._compute_binary_diff(source_content, target_content)
                
                # Save diff to file
                diff_data = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "patches": [patch.model_dump() for patch in patches]
                }
                safe_save_json(diff_data, diff_file)
        
        return SnapshotDiff(
            source_id=source_id,
            target_id=target_id,
            added_lines=added_lines,
            removed_lines=removed_lines,
            changed_lines=changed_lines,
            diff_content='\n'.join(diff),
            patches=patches
        )
    
    def create_branch(self, name: str, base_branch: str = "main", protected: bool = False, description: Optional[str] = None) -> Branch:
        """
        Create a new branch.
        
        Args:
            name: Name of the branch
            base_branch: Branch to base the new branch on
            protected: Whether the branch is protected from deletion
            description: Optional description of the branch
            
        Returns:
            The created branch
        """
        if name in self.branches:
            return self.branches[name]
            
        head_id = None
        if base_branch in self.branches and self.branches[base_branch].head_snapshot_id:
            head_id = self.branches[base_branch].head_snapshot_id
            
        branch = Branch(
            name=name,
            head_snapshot_id=head_id,
            protected=protected,
            description=description
        )
        
        if head_id:
            branch.snapshots.append(head_id)
            
        self.branches[name] = branch
        self._save_branches()
        
        return branch
    
    def delete_branch(self, name: str, force: bool = False) -> bool:
        """
        Delete a branch.
        
        Args:
            name: Name of the branch to delete
            force: Whether to force deletion of a protected branch
            
        Returns:
            Whether the deletion was successful
        """
        if name not in self.branches:
            return False
        
        branch = self.branches[name]
        
        # Don't delete main branch
        if name == "main":
            logger.warning("Cannot delete main branch")
            return False
        
        # Don't delete protected branches unless forced
        if branch.protected and not force:
            logger.warning(f"Branch {name} is protected. Use force=True to delete")
            return False
        
        # Remove the branch
        del self.branches[name]
        self._save_branches()
        
        return True
    
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
            metadata = self.get_snapshot_metadata(snapshot_id)
            if metadata:
                snapshots.append(metadata)
        
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
            
        metadata = self.get_snapshot_metadata(snapshot_id)
        if metadata is None:
            return None
        
        # Determine output path
        if output_path is None:
            output_path = resolve_path(metadata.file_path)
            
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Write content
        with open(output_path, 'w') as f:
            f.write(content)
            
        return str(output_path)
    
    def rollback_to_snapshot(self, snapshot_id: str, branch: Optional[str] = None, message: Optional[str] = None) -> Optional[SnapshotMetadata]:
        """
        Rollback to a previous snapshot by creating a new snapshot with the same content.
        
        Args:
            snapshot_id: ID of the snapshot to rollback to
            branch: Optional branch to rollback on (defaults to the snapshot's branch)
            message: Optional message for the rollback snapshot
            
        Returns:
            Metadata for the rollback snapshot or None if the source snapshot was not found
        """
        # Get the snapshot to rollback to
        content = self.get_snapshot_content(snapshot_id)
        if content is None:
            logger.error(f"Snapshot {snapshot_id} not found")
            return None
        
        # Get metadata for the snapshot
        metadata = self.get_snapshot_metadata(snapshot_id)
        if metadata is None:
            logger.error(f"Metadata for snapshot {snapshot_id} not found")
            return None
        
        # Determine branch
        rollback_branch = branch or "main"
        
        # Determine message
        rollback_message = message or f"Rollback to snapshot {snapshot_id}"
        
        # Create a new snapshot with the rollback content
        try:
            rollback_metadata = self.create_snapshot(
                file_path=metadata.file_path,
                content=content,
                branch=rollback_branch,
                message=rollback_message,
                tags=metadata.tags + ["rollback"],
                status="success",
                language=metadata.language
            )
            
            logger.info(f"Rolled back to snapshot {snapshot_id} by creating new snapshot {rollback_metadata.snapshot_id}")
            return rollback_metadata
        except Exception as e:
            logger.error(f"Failed to rollback to snapshot {snapshot_id}: {e}")
            return None
    
    def revert_snapshot(self, snapshot_id: str) -> Optional[SnapshotMetadata]:
        """
        Mark a snapshot as reverted without creating a new snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to revert
            
        Returns:
            Updated metadata for the reverted snapshot or None if not found
        """
        metadata = self.get_snapshot_metadata(snapshot_id)
        if metadata is None:
            return None
        
        # Update status to reverted
        metadata.status = "reverted"
        
        # Save updated metadata
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        success = safe_save_json(metadata.model_dump(), metadata_path)
        if not success:
            logger.error(f"Failed to update metadata for snapshot {snapshot_id}")
            return None
        
        # Update cache
        if snapshot_id in self._metadata_cache:
            self._metadata_cache[snapshot_id] = metadata
        
        logger.info(f"Marked snapshot {snapshot_id} as reverted")
        return metadata
    
    def get_snapshot_chain(self, snapshot_id: str, max_depth: int = 10) -> List[SnapshotMetadata]:
        """
        Get a chain of snapshots leading up to the given snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to get the chain for
            max_depth: Maximum depth of the chain
            
        Returns:
            List of snapshot metadata in ancestry order (newest first)
        """
        chain = []
        current_id = snapshot_id
        depth = 0
        
        while current_id and depth < max_depth:
            metadata = self.get_snapshot_metadata(current_id)
            if metadata is None:
                break
                
            chain.append(metadata)
            current_id = metadata.parent_snapshot_id
            depth += 1
        
        return chain
    
    def find_common_ancestor(self, snapshot_id1: str, snapshot_id2: str) -> Optional[str]:
        """
        Find the common ancestor of two snapshots.
        
        Args:
            snapshot_id1: ID of the first snapshot
            snapshot_id2: ID of the second snapshot
            
        Returns:
            ID of the common ancestor or None if not found
        """
        # Get chains for both snapshots
        chain1 = [meta.snapshot_id for meta in self.get_snapshot_chain(snapshot_id1)]
        chain2 = [meta.snapshot_id for meta in self.get_snapshot_chain(snapshot_id2)]
        
        # Find common ancestors
        for snapshot_id in chain1:
            if snapshot_id in chain2:
                return snapshot_id
        
        return None
    
    def get_file_history(self, file_path: Union[str, Path], branch: Optional[str] = None) -> List[SnapshotMetadata]:
        """
        Get the history of a file.
        
        Args:
            file_path: Path to the file
            branch: Optional branch to filter by
            
        Returns:
            List of snapshot metadata in chronological order
        """
        snapshots = self.get_snapshots(file_path, branch)
        return sorted(snapshots, key=lambda x: x.timestamp)
    
    def get_latest_successful_snapshot(self, file_path: Union[str, Path], branch: str = "main") -> Optional[SnapshotMetadata]:
        """
        Get the latest successful snapshot for a file.
        
        Args:
            file_path: Path to the file
            branch: Branch to get the snapshot from
            
        Returns:
            Latest successful snapshot metadata or None if not found
        """
        snapshots = self.get_snapshots(file_path, branch)
        
        # Filter for successful snapshots
        successful = [s for s in snapshots if s.status == "success"]
        
        return successful[0] if successful else None
    
    def export_version_summary(self, file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Export a summary of all versions of a file.
        
        Args:
            file_path: Path to the file
            output_path: Optional path to export to
            
        Returns:
            Path to the exported summary or None if no snapshots found
        """
        snapshots = self.get_snapshots(file_path)
        if not snapshots:
            return None
        
        # Generate summary
        summary = f"# Version History for {Path(file_path).name}\n\n"
        
        for i, snapshot in enumerate(snapshots):
            summary += f"## Version {len(snapshots)-i}: {snapshot.snapshot_id}\n"
            summary += f"- Timestamp: {snapshot.timestamp}\n"
            summary += f"- Author: {snapshot.author}\n"
            summary += f"- Status: {snapshot.status}\n"
            
            if snapshot.message:
                summary += f"- Message: {snapshot.message}\n"
                
            if snapshot.tags:
                summary += f"- Tags: {', '.join(snapshot.tags)}\n"
                
            if i < len(snapshots) - 1 and snapshots[i+1].snapshot_id:
                diff = self.compare_snapshots(snapshots[i+1].snapshot_id, snapshot.snapshot_id, compute_binary_diff=False)
                if diff:
                    summary += f"- Changes: {diff.summary()}\n"
            
            summary += "\n"
        
        # Determine output path
        if output_path is None:
            output_path = resolve_path(f"artifacts/summaries/{Path(file_path).name}_summary.md", create_parents=True)
        else:
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)
        
        # Write summary
        with open(output_path, 'w') as f:
            f.write(summary)
            
        return str(output_path)
    
    def get_snapshot_size_info(self) -> Dict[str, Any]:
        """
        Get information about snapshot sizes.
        
        Returns:
            Dictionary with snapshot size statistics
        """
        sizes = []
        total_size = 0
        largest_snapshot_id = ""
        largest_size = 0
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                data = load_json(metadata_file)
                metadata = SnapshotMetadata.model_validate(data)
                
                # Get size from metadata or compute it
                size = metadata.size
                if size == 0:
                    content = self.get_snapshot_content(metadata.snapshot_id)
                    if content:
                        size = len(content)
                        
                        # Update the metadata with the correct size
                        metadata.size = size
                        safe_save_json(metadata.model_dump(), metadata_file)
                
                sizes.append(size)
                total_size += size
                
                if size > largest_size:
                    largest_size = size
                    largest_snapshot_id = metadata.snapshot_id
                    
            except Exception as e:
                logger.error(f"Error processing metadata file {metadata_file}: {e}")
        
        # Compute statistics
        return {
            "total_snapshots": len(sizes),
            "total_size_bytes": total_size,
            "average_size_bytes": total_size / len(sizes) if sizes else 0,
            "largest_snapshot_id": largest_snapshot_id,
            "largest_snapshot_size_bytes": largest_size,
            "median_size_bytes": sorted(sizes)[len(sizes)//2] if sizes else 0
        }
    
    def tag_snapshot(self, snapshot_id: str, tags: List[str]) -> bool:
        """
        Add tags to a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to tag
            tags: Tags to add
            
        Returns:
            Whether the update was successful
        """
        metadata = self.get_snapshot_metadata(snapshot_id)
        if metadata is None:
            return False
        
        # Update tags, ensuring uniqueness
        metadata.tags = list(set(metadata.tags + tags))
        
        # Save updated metadata
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        success = safe_save_json(metadata.model_dump(), metadata_path)
        
        # Update cache
        if snapshot_id in self._metadata_cache:
            self._metadata_cache[snapshot_id] = metadata
        
        return success