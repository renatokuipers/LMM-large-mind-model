# src/agendev/UI/snapshot_integration.py
"""Integration between the snapshot engine and UI components."""

import html
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import dash
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..snapshot_engine import SnapshotEngine, SnapshotMetadata
from .snapshot_components import (
    create_snapshot_management_view,
    create_code_explorer_with_history,
    create_snapshot_controls,
    create_snapshot_diff_view
)

logger = logging.getLogger(__name__)

# Global instance of the snapshot engine
snapshot_engine = SnapshotEngine()

def register_snapshot_callbacks(app: dash.Dash) -> None:
    """
    Register all callbacks related to snapshot functionality.
    
    Args:
        app: Dash application instance
    """
    # Register snapshot filter callbacks
    @app.callback(
        Output("snapshot-management-container", "children"),
        [
            Input("snapshot-apply-filters", "n_clicks"),
            Input("snapshot-file-filter", "value"),
            Input("snapshot-branch-filter", "value"),
            Input("snapshot-status-filter", "value")
        ],
        prevent_initial_call=True
    )
    def update_snapshot_filters(n_clicks, file_filter, branch_filter, status_filter):
        """Update snapshot filters."""
        if not n_clicks and not callback_context.triggered:
            raise PreventUpdate
        
        # Filter snapshots
        snapshots = []
        if file_filter:
            snapshots = snapshot_engine.get_snapshots(file_filter, branch_filter)
        else:
            snapshots = snapshot_engine.get_all_snapshots(branch_filter)
        
        # Apply status filter if specified
        if status_filter:
            snapshots = [s for s in snapshots if s.status == status_filter]
        
        # Create updated view
        return create_snapshot_management_view(
            snapshot_engine, 
            file_path=file_filter, 
            branch=branch_filter
        )
    
    # Register snapshot selection callback
    @app.callback(
        [
            Output("snapshot-details-container", "children"),
            Output("selected-snapshot-id", "data"),
            Output("snapshot-tabs", "active_tab"),
            Output("snapshot-tabs", "children")
        ],
        [Input("snapshot-history-table", "selected_rows")],
        [State("snapshot-history-table", "data")]
    )
    def select_snapshot(selected_rows, table_data):
        """Handle snapshot selection."""
        if not selected_rows or not table_data:
            raise PreventUpdate
        
        # Get selected snapshot ID
        row_idx = selected_rows[0]
        snapshot_id = table_data[row_idx]["id"]
        
        # Create controls for the selected snapshot
        controls = create_snapshot_controls(snapshot_id, snapshot_engine)
        
        # Update the tabs
        tabs = dash.callback_context.outputs_list[3]
        tabs[2]["props"]["disabled"] = False  # Enable details tab
        
        return controls, snapshot_id, "details-tab", tabs
    
    # Register compare with parent callback
    @app.callback(
        [
            Output("snapshot-diff-container", "children"),
            Output("snapshot-tabs", "active_tab", allow_duplicate=True),
            Output("comparison-source-id", "data"),
            Output("comparison-target-id", "data")
        ],
        [Input({"type": "snapshot-compare-parent-button", "index": dash.ALL}, "n_clicks")],
        [State("selected-snapshot-id", "data")],
        prevent_initial_call=True
    )
    def compare_with_parent(n_clicks_list, snapshot_id):
        """Compare selected snapshot with its parent."""
        if not n_clicks_list or not any(n_clicks_list) or not snapshot_id:
            raise PreventUpdate
        
        # Get metadata for the snapshot
        metadata = snapshot_engine.get_snapshot_metadata(snapshot_id)
        if metadata is None or metadata.parent_snapshot_id is None:
            return "No parent snapshot found", "details-tab", None, None
        
        # Get parent metadata
        parent_id = metadata.parent_snapshot_id
        parent_metadata = snapshot_engine.get_snapshot_metadata(parent_id)
        if parent_metadata is None:
            return "Parent metadata not found", "details-tab", None, None
        
        # Compare snapshots
        diff = snapshot_engine.compare_snapshots(parent_id, snapshot_id)
        if diff is None:
            return "Failed to generate diff", "details-tab", None, None
        
        # Create diff view
        diff_view = create_snapshot_diff_view(diff, parent_metadata, metadata)
        
        return diff_view, "diff-tab", parent_id, snapshot_id
    
    # Register snapshot restore callback
    @app.callback(
        Output("snapshot-restore-notification", "children"),
        [Input({"type": "snapshot-restore-button", "index": dash.ALL}, "n_clicks")],
        [State("selected-snapshot-id", "data")],
        prevent_initial_call=True
    )
    def restore_snapshot(n_clicks_list, snapshot_id):
        """Restore selected snapshot to a file."""
        if not n_clicks_list or not any(n_clicks_list) or not snapshot_id:
            raise PreventUpdate
        
        try:
            # Restore the snapshot
            output_path = snapshot_engine.restore_snapshot(snapshot_id)
            if output_path:
                return dbc.Alert(
                    f"Snapshot {snapshot_id} restored to {output_path}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
            else:
                return dbc.Alert(
                    f"Failed to restore snapshot {snapshot_id}",
                    color="danger",
                    dismissable=True,
                    duration=4000
                )
        except Exception as e:
            logger.error(f"Error restoring snapshot {snapshot_id}: {e}")
            return dbc.Alert(
                f"Error: {str(e)}",
                color="danger",
                dismissable=True,
                duration=4000
            )
    
    # Register tag modal callback
    @app.callback(
        [
            Output("tag-modal", "is_open"),
            Output("tag-modal", "children")
        ],
        [
            Input({"type": "snapshot-tag-button", "index": dash.ALL}, "n_clicks"),
            Input("tag-modal-cancel", "n_clicks"),
            Input({"type": "tag-modal-submit", "index": dash.ALL}, "n_clicks")
        ],
        [
            State("selected-snapshot-id", "data"),
            State("tag-modal", "is_open"),
            State("tag-input-field", "value")
        ],
        prevent_initial_call=True
    )
    def handle_tag_modal(tag_clicks, cancel_clicks, submit_clicks, snapshot_id, is_open, tag_input):
        """Handle tag modal open/close and submission."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Open the modal
        if trigger_id.startswith("{\"type\":\"snapshot-tag-button\"") and not is_open and snapshot_id:
            from .snapshot_components import create_tag_input_modal
            return True, create_tag_input_modal(snapshot_id)
        
        # Cancel the modal
        if trigger_id == "tag-modal-cancel" and is_open:
            return False, dash.no_update
        
        # Submit the tags
        if trigger_id.startswith("{\"type\":\"tag-modal-submit\"") and is_open and tag_input:
            # Parse tags
            tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
            
            if tags and snapshot_id:
                # Add tags to the snapshot
                success = snapshot_engine.tag_snapshot(snapshot_id, tags)
                if success:
                    # Close the modal
                    return False, dash.no_update
        
        # Default case
        return is_open, dash.no_update
    
    # Register rollback modal callback
    @app.callback(
        [
            Output("rollback-modal", "is_open"),
            Output("rollback-modal", "children")
        ],
        [
            Input({"type": "snapshot-rollback-button", "index": dash.ALL}, "n_clicks"),
            Input("rollback-modal-cancel", "n_clicks"),
            Input({"type": "rollback-modal-submit", "index": dash.ALL}, "n_clicks")
        ],
        [
            State("selected-snapshot-id", "data"),
            State("rollback-modal", "is_open"),
            State("rollback-message-input", "value")
        ],
        prevent_initial_call=True
    )
    def handle_rollback_modal(rollback_clicks, cancel_clicks, submit_clicks, snapshot_id, is_open, message):
        """Handle rollback modal open/close and submission."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Open the modal
        if trigger_id.startswith("{\"type\":\"snapshot-rollback-button\"") and not is_open and snapshot_id:
            from .snapshot_components import create_rollback_confirm_modal
            return True, create_rollback_confirm_modal(snapshot_id)
        
        # Cancel the modal
        if trigger_id == "rollback-modal-cancel" and is_open:
            return False, dash.no_update
        
        # Submit the rollback
        if trigger_id.startswith("{\"type\":\"rollback-modal-submit\"") and is_open:
            if snapshot_id:
                # Perform the rollback
                rollback_metadata = snapshot_engine.rollback_to_snapshot(
                    snapshot_id, 
                    message=message if message else f"Rollback to {snapshot_id}"
                )
                
                if rollback_metadata:
                    # Close the modal
                    return False, dash.no_update
        
        # Default case
        return is_open, dash.no_update
    
    # Register revert snapshot callback
    @app.callback(
        Output("snapshot-revert-notification", "children"),
        [Input({"type": "snapshot-revert-button", "index": dash.ALL}, "n_clicks")],
        [State("selected-snapshot-id", "data")],
        prevent_initial_call=True
    )
    def revert_snapshot(n_clicks_list, snapshot_id):
        """Mark a snapshot as reverted."""
        if not n_clicks_list or not any(n_clicks_list) or not snapshot_id:
            raise PreventUpdate
        
        try:
            # Mark the snapshot as reverted
            updated_metadata = snapshot_engine.revert_snapshot(snapshot_id)
            if updated_metadata:
                return dbc.Alert(
                    f"Snapshot {snapshot_id} marked as reverted",
                    color="warning",
                    dismissable=True,
                    duration=4000
                )
            else:
                return dbc.Alert(
                    f"Failed to mark snapshot {snapshot_id} as reverted",
                    color="danger",
                    dismissable=True,
                    duration=4000
                )
        except Exception as e:
            logger.error(f"Error reverting snapshot {snapshot_id}: {e}")
            return dbc.Alert(
                f"Error: {str(e)}",
                color="danger",
                dismissable=True,
                duration=4000
            )
    
    # Register snapshot navigation callbacks for code explorer
    @app.callback(
        [
            Output("code-explorer-content", "children"),
            Output("code-explorer-snapshot-selector", "value")
        ],
        [
            Input("code-explorer-snapshot-selector", "value"),
            Input("code-explorer-prev-button", "n_clicks"),
            Input("code-explorer-next-button", "n_clicks"),
            Input("code-explorer-latest-button", "n_clicks")
        ],
        [
            State("code-explorer-snapshot-list", "data"),
            State("code-explorer-snapshot-selector", "value")
        ],
        prevent_initial_call=True
    )
    def navigate_code_explorer(selected_id, prev_clicks, next_clicks, latest_clicks, snapshot_list, current_id):
        """Handle navigation in code explorer."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Determine which snapshot to show
        target_id = current_id
        
        if trigger_id == "code-explorer-snapshot-selector":
            target_id = selected_id
        elif trigger_id == "code-explorer-prev-button" and current_id in snapshot_list:
            idx = snapshot_list.index(current_id)
            if idx < len(snapshot_list) - 1:
                target_id = snapshot_list[idx + 1]
        elif trigger_id == "code-explorer-next-button" and current_id in snapshot_list:
            idx = snapshot_list.index(current_id)
            if idx > 0:
                target_id = snapshot_list[idx - 1]
        elif trigger_id == "code-explorer-latest-button" and snapshot_list:
            target_id = snapshot_list[0]
        
        # Get snapshot content
        content = snapshot_engine.get_snapshot_content(target_id) or ""
        
        return html.Pre(content), target_id
    
    # Register code comparison callback
    @app.callback(
        Output("code-explorer-comparison-result", "children"),
        [Input("code-explorer-compare-button", "n_clicks")],
        [
            State("code-explorer-snapshot-selector", "value"),
            State("code-explorer-compare-selector", "value")
        ],
        prevent_initial_call=True
    )
    def compare_code_versions(n_clicks, source_id, target_id):
        """Compare two code versions."""
        if not n_clicks or not source_id or not target_id:
            raise PreventUpdate
        
        # Get metadata
        source_metadata = snapshot_engine.get_snapshot_metadata(source_id)
        target_metadata = snapshot_engine.get_snapshot_metadata(target_id)
        
        if not source_metadata or not target_metadata:
            return html.Div("Failed to find snapshot metadata", className="text-danger")
        
        # Compare snapshots
        diff = snapshot_engine.compare_snapshots(source_id, target_id)
        if diff is None:
            return html.Div("Failed to generate diff", className="text-danger")
        
        # Create diff view
        return create_snapshot_diff_view(diff, source_metadata, target_metadata)
    
    # Register snapshot restoration callback for code explorer
    @app.callback(
        Output("code-explorer-restore-notification", "children"),
        [Input("code-explorer-restore-button", "n_clicks")],
        [
            State("code-explorer-snapshot-selector", "value"),
            State("code-explorer-file-path", "data")
        ],
        prevent_initial_call=True
    )
    def restore_from_code_explorer(n_clicks, snapshot_id, file_path):
        """Restore a snapshot from the code explorer."""
        if not n_clicks or not snapshot_id or not file_path:
            raise PreventUpdate
        
        try:
            # Restore the snapshot
            output_path = snapshot_engine.restore_snapshot(snapshot_id, file_path)
            if output_path:
                return dbc.Alert(
                    f"Snapshot {snapshot_id} restored to {output_path}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
            else:
                return dbc.Alert(
                    f"Failed to restore snapshot",
                    color="danger",
                    dismissable=True,
                    duration=4000
                )
        except Exception as e:
            logger.error(f"Error restoring snapshot: {e}")
            return dbc.Alert(
                f"Error: {str(e)}",
                color="danger",
                dismissable=True,
                duration=4000
            )

def get_snapshot_engine() -> SnapshotEngine:
    """
    Get the global snapshot engine instance.
    
    Returns:
        Snapshot engine instance
    """
    return snapshot_engine

def snapshot_file(file_path: Union[str, Path], content: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> SnapshotMetadata:
    """
    Create a snapshot of a file (convenience function).
    
    Args:
        file_path: Path to the file
        content: Content of the file
        metadata: Optional additional metadata
        
    Returns:
        Metadata for the created snapshot
    """
    if metadata is None:
        metadata = {}
    
    # Extract metadata fields
    branch = metadata.get("branch", "main")
    message = metadata.get("message")
    tags = metadata.get("tags", [])
    status = metadata.get("status", "success")
    language = metadata.get("language")
    execution_time = metadata.get("execution_time")
    compress = metadata.get("compress", False)
    
    # Create snapshot
    return snapshot_engine.create_snapshot(
        file_path=file_path,
        content=content,
        branch=branch,
        message=message,
        tags=tags,
        status=status,
        language=language,
        execution_time=execution_time,
        compress=compress
    )

def rollback_file(file_path: Union[str, Path], snapshot_id: Optional[str] = None) -> Optional[str]:
    """
    Rollback a file to a previous snapshot.
    
    Args:
        file_path: Path to the file
        snapshot_id: Optional specific snapshot ID to rollback to 
                    (if None, uses latest successful snapshot)
        
    Returns:
        Content of the rollback snapshot or None if failed
    """
    # Determine which snapshot to roll back to
    target_id = snapshot_id
    if target_id is None:
        # Find the latest successful snapshot
        snapshots = snapshot_engine.get_snapshots(file_path)
        successful = [s for s in snapshots if s.status == "success"]
        if not successful:
            logger.error(f"No successful snapshots found for {file_path}")
            return None
        target_id = successful[0].snapshot_id
    
    # Get the snapshot content
    content = snapshot_engine.get_snapshot_content(target_id)
    if content is None:
        logger.error(f"Failed to get content for snapshot {target_id}")
        return None
    
    # Create a rollback snapshot
    metadata = snapshot_engine.rollback_to_snapshot(
        snapshot_id=target_id,
        message=f"Rollback to snapshot {target_id}"
    )
    
    if metadata is None:
        logger.error(f"Failed to create rollback snapshot for {file_path}")
        return None
    
    logger.info(f"Rolled back {file_path} to snapshot {target_id}")
    return content

def mark_snapshot_failed(snapshot_id: str, error_message: Optional[str] = None) -> bool:
    """
    Mark a snapshot as failed.
    
    Args:
        snapshot_id: ID of the snapshot to mark as failed
        error_message: Optional error message to include
        
    Returns:
        Whether the update was successful
    """
    metadata = snapshot_engine.get_snapshot_metadata(snapshot_id)
    if metadata is None:
        logger.error(f"Failed to find metadata for snapshot {snapshot_id}")
        return False
    
    # Update status
    metadata.status = "failed"
    
    # Update message if provided
    if error_message:
        if metadata.message:
            metadata.message = f"{metadata.message} - Error: {error_message}"
        else:
            metadata.message = f"Error: {error_message}"
    
    # Add error tag
    if "error" not in metadata.tags:
        metadata.tags.append("error")
    
    # Save updated metadata
    metadata_path = snapshot_engine.metadata_dir / f"{snapshot_id}.json"
    from ..utils.fs_utils import safe_save_json
    success = safe_save_json(metadata.model_dump(), metadata_path)
    
    # Update cache
    if snapshot_id in snapshot_engine._metadata_cache:
        snapshot_engine._metadata_cache[snapshot_id] = metadata
    
    logger.info(f"Marked snapshot {snapshot_id} as failed")
    return success

def get_version_history(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Get a simplified version history for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of dictionaries with version information
    """
    snapshots = snapshot_engine.get_snapshots(file_path)
    
    history = []
    for s in snapshots:
        version_info = {
            "id": s.snapshot_id,
            "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, "isoformat") else s.timestamp,
            "message": s.message,
            "status": s.status,
            "tags": s.tags
        }
        history.append(version_info)
    
    return history

def analyze_snapshots(file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Analyze snapshots to gain insights.
    
    Args:
        file_path: Optional path to filter snapshots
        
    Returns:
        Dictionary with analysis results
    """
    if file_path:
        snapshots = snapshot_engine.get_snapshots(file_path)
    else:
        snapshots = snapshot_engine.get_all_snapshots()
    
    # Sort by timestamp
    snapshots = sorted(snapshots, key=lambda x: x.timestamp)
    
    # Analyze status distribution
    status_counts = {
        "success": 0,
        "failed": 0,
        "reverted": 0
    }
    
    for s in snapshots:
        if s.status in status_counts:
            status_counts[s.status] += 1
    
    # Analyze file types
    file_types = {}
    for s in snapshots:
        ext = Path(s.file_path).suffix.lstrip('.')
        if ext:
            if ext not in file_types:
                file_types[ext] = 0
            file_types[ext] += 1
    
    # Analyze size evolution if we have multiple snapshots for the same file
    size_evolution = {}
    if file_path:
        for s in snapshots:
            if s.size > 0:
                ts = s.timestamp.isoformat() if hasattr(s.timestamp, "isoformat") else s.timestamp
                size_evolution[ts] = s.size
    
    # Analyze snapshot frequency
    snapshot_dates = [s.timestamp for s in snapshots]
    date_counts = {}
    for date in snapshot_dates:
        date_str = date.date().isoformat() if hasattr(date, "date") else date.split("T")[0]
        if date_str not in date_counts:
            date_counts[date_str] = 0
        date_counts[date_str] += 1
    
    # Compile results
    return {
        "total_snapshots": len(snapshots),
        "status_distribution": status_counts,
        "file_types": file_types,
        "size_evolution": size_evolution,
        "snapshot_frequency": date_counts,
        "first_snapshot": snapshots[0].timestamp if snapshots else None,
        "last_snapshot": snapshots[-1].timestamp if snapshots else None
    }