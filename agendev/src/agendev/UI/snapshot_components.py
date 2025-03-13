# src/agendev/UI/snapshot_components.py
"""UI components for interacting with snapshots."""

import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import html, dcc, dash_table
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import difflib
from datetime import datetime
import plotly.graph_objects as go

from ..snapshot_engine import SnapshotEngine, SnapshotMetadata, SnapshotDiff


def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """Format a timestamp for display."""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except:
            return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def create_snapshot_history_table(snapshots: List[SnapshotMetadata], current_id: Optional[str] = None) -> html.Div:
    """
    Create a table showing snapshot history.
    
    Args:
        snapshots: List of snapshots to display
        current_id: Currently selected snapshot ID
        
    Returns:
        Dash component with the table
    """
    if not snapshots:
        return html.Div("No snapshots available", className="text-center text-muted my-3")
    
    # Convert to records for the data table
    records = []
    for s in snapshots:
        record = {
            "id": s.snapshot_id,
            "timestamp": format_timestamp(s.timestamp),
            "message": s.message or "",
            "status": s.status,
            "tags": ", ".join(s.tags) if s.tags else "",
            "size": f"{s.size:,} bytes" if s.size else "Unknown"
        }
        records.append(record)
    
    # Create table columns
    columns = [
        {"name": "ID", "id": "id"},
        {"name": "Timestamp", "id": "timestamp"},
        {"name": "Message", "id": "message"},
        {"name": "Status", "id": "status"},
        {"name": "Tags", "id": "tags"},
        {"name": "Size", "id": "size"}
    ]
    
    # Style for the active row
    style_data_conditional = []
    if current_id:
        style_data_conditional.append({
            "if": {"filter_query": f"{{id}} = '{current_id}'"},
            "backgroundColor": "#4E5D6C",
            "color": "white"
        })
    
    # Style for status column
    style_data_conditional.extend([
        {
            "if": {"filter_query": "{status} = 'success'", "column_id": "status"},
            "backgroundColor": "#5cb85c",
            "color": "white"
        },
        {
            "if": {"filter_query": "{status} = 'failed'", "column_id": "status"},
            "backgroundColor": "#d9534f",
            "color": "white"
        },
        {
            "if": {"filter_query": "{status} = 'reverted'", "column_id": "status"},
            "backgroundColor": "#f0ad4e",
            "color": "white"
        }
    ])
    
    return html.Div([
        dash_table.DataTable(
            id="snapshot-history-table",
            columns=columns,
            data=records,
            page_size=10,
            style_header={
                "backgroundColor": "#2d2d2d",
                "color": "white",
                "fontWeight": "bold"
            },
            style_cell={
                "backgroundColor": "#3d3d3d",
                "color": "#ccc",
                "textAlign": "left",
                "padding": "10px",
                "whiteSpace": "normal",
                "height": "auto"
            },
            style_data_conditional=style_data_conditional,
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            selected_rows=[0] if records else [],
            tooltip_data=[
                {
                    "id": {"value": f"Click to select snapshot {row['id']}", "type": "markdown"},
                }
                for row in records
            ],
            css=[
                {"selector": ".dash-table-tooltip", "rule": "background-color: #2d2d2d; color: white;"}
            ]
        )
    ], className="mb-4")


def create_snapshot_diff_view(diff: SnapshotDiff, source_metadata: Optional[SnapshotMetadata] = None, 
                             target_metadata: Optional[SnapshotMetadata] = None) -> html.Div:
    """
    Create a component to visualize differences between snapshots.
    
    Args:
        diff: The snapshot diff to display
        source_metadata: Metadata for the source snapshot
        target_metadata: Metadata for the target snapshot
        
    Returns:
        Dash component with the diff visualization
    """
    # Format source and target metadata if available
    source_info = ""
    target_info = ""
    
    if source_metadata:
        source_info = f"From: {source_metadata.snapshot_id} ({format_timestamp(source_metadata.timestamp)})"
        if source_metadata.message:
            source_info += f" - {source_metadata.message}"
    
    if target_metadata:
        target_info = f"To: {target_metadata.snapshot_id} ({format_timestamp(target_metadata.timestamp)})"
        if target_metadata.message:
            target_info += f" - {target_metadata.message}"
    
    # Format diff lines with syntax highlighting
    diff_lines = diff.diff_content.splitlines()
    formatted_diff = []
    
    for line in diff_lines:
        if line.startswith('+'):
            # Added line
            formatted_diff.append(html.Div(
                line, 
                style={"backgroundColor": "#264026", "color": "#5cb85c"}
            ))
        elif line.startswith('-'):
            # Removed line
            formatted_diff.append(html.Div(
                line, 
                style={"backgroundColor": "#402626", "color": "#d9534f"}
            ))
        elif line.startswith('@'):
            # Chunk header
            formatted_diff.append(html.Div(
                line, 
                style={"backgroundColor": "#262640", "color": "#5bc0de"}
            ))
        else:
            # Context line
            formatted_diff.append(html.Div(line, style={"color": "#ccc"}))
    
    # Create summary card
    summary_card = dbc.Card([
        dbc.CardHeader("Changes Summary"),
        dbc.CardBody([
            html.Div([
                html.Div(source_info, className="text-muted"),
                html.Div(target_info, className="text-muted"),
                html.Hr(),
                html.Div([
                    html.Span(f"Added: ", className="mr-2"),
                    html.Span(
                        f"{diff.added_lines}", 
                        className="badge bg-success"
                    ),
                    html.Span(f"  Removed: ", className="mx-2"),
                    html.Span(
                        f"{diff.removed_lines}", 
                        className="badge bg-danger"
                    ),
                    html.Span(f"  Changed: ", className="mx-2"),
                    html.Span(
                        f"{diff.changed_lines}", 
                        className="badge bg-info"
                    ),
                ], className="d-flex align-items-center")
            ])
        ])
    ], className="mb-3")
    
    # Create diff viewer
    diff_viewer = dbc.Card([
        dbc.CardHeader("Diff Details"),
        dbc.CardBody([
            html.Div(
                formatted_diff,
                className="p-3",
                style={
                    "backgroundColor": "#1e1e1e",
                    "fontFamily": "monospace",
                    "whiteSpace": "pre",
                    "overflowX": "auto",
                    "borderRadius": "4px"
                }
            )
        ])
    ])
    
    return html.Div([
        summary_card,
        diff_viewer
    ])


def create_snapshot_visualization(snapshots: List[SnapshotMetadata]) -> html.Div:
    """
    Create a visual graph of snapshot history.
    
    Args:
        snapshots: List of snapshots to visualize
        
    Returns:
        Dash component with the visualization
    """
    if not snapshots:
        return html.Div("No snapshots available for visualization", className="text-muted text-center my-3")
    
    # Sort snapshots by timestamp
    sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)
    
    # Create data for the timeline
    timestamps = [s.timestamp for s in sorted_snapshots]
    ids = [s.snapshot_id for s in sorted_snapshots]
    messages = [s.message or f"Snapshot {s.snapshot_id}" for s in sorted_snapshots]
    statuses = [s.status for s in sorted_snapshots]
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for status
    color_map = {
        "success": "#5cb85c",
        "failed": "#d9534f",
        "reverted": "#f0ad4e"
    }
    
    # Add points for each snapshot
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[1] * len(timestamps),
        mode="markers+text",
        marker=dict(
            size=20,
            color=[color_map.get(status, "#ccc") for status in statuses],
            line=dict(width=2, color="white")
        ),
        text=ids,
        textposition="top center",
        name="Snapshots",
        hovertext=[f"ID: {id}<br>Time: {format_timestamp(ts)}<br>Message: {msg}" 
                  for id, ts, msg in zip(ids, timestamps, messages)],
        hoverinfo="text"
    ))
    
    # Add connecting lines
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[1] * len(timestamps),
        mode="lines",
        line=dict(color="#ccc", width=2),
        showlegend=False
    ))
    
    # Set layout
    fig.update_layout(
        title="Snapshot Timeline",
        xaxis_title="Time",
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="#2d2d2d",
        paper_bgcolor="#2d2d2d",
        font=dict(color="#ccc"),
        hovermode="closest",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return html.Div([
        dcc.Graph(
            id="snapshot-timeline-graph",
            figure=fig,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )
    ], className="mb-4")


def create_snapshot_controls(snapshot_id: str, engine: SnapshotEngine) -> html.Div:
    """
    Create controls for interacting with a snapshot.
    
    Args:
        snapshot_id: ID of the current snapshot
        engine: Snapshot engine instance
        
    Returns:
        Dash component with snapshot controls
    """
    # Get metadata for the snapshot
    metadata = engine.get_snapshot_metadata(snapshot_id)
    if metadata is None:
        return html.Div("Snapshot not found", className="text-danger")
    
    # Get parent metadata if available
    parent_metadata = None
    if metadata.parent_snapshot_id:
        parent_metadata = engine.get_snapshot_metadata(metadata.parent_snapshot_id)
    
    # Get next snapshot if available (snapshots that have this as a parent)
    next_snapshots = []
    all_snapshots = engine.get_all_snapshots()
    for s in all_snapshots:
        if s.parent_snapshot_id == snapshot_id:
            next_snapshots.append(s)
    
    # Create info card
    info_card = dbc.Card([
        dbc.CardHeader("Snapshot Information"),
        dbc.CardBody([
            html.Div([
                html.Strong("ID: "),
                html.Span(metadata.snapshot_id)
            ], className="mb-2"),
            html.Div([
                html.Strong("File: "),
                html.Span(metadata.file_path)
            ], className="mb-2"),
            html.Div([
                html.Strong("Created: "),
                html.Span(format_timestamp(metadata.timestamp))
            ], className="mb-2"),
            html.Div([
                html.Strong("Status: "),
                html.Span(
                    metadata.status,
                    className=f"badge {'bg-success' if metadata.status == 'success' else 'bg-danger' if metadata.status == 'failed' else 'bg-warning'}"
                )
            ], className="mb-2"),
            html.Div([
                html.Strong("Size: "),
                html.Span(f"{metadata.size:,} bytes" if metadata.size else "Unknown")
            ], className="mb-2"),
            html.Div([
                html.Strong("Message: "),
                html.Span(metadata.message or "N/A")
            ], className="mb-2"),
            html.Div([
                html.Strong("Tags: "),
                html.Span(", ".join(metadata.tags) if metadata.tags else "None")
            ], className="mb-2"),
            html.Div([
                html.Strong("Language: "),
                html.Span(metadata.language or "Not specified")
            ], className="mb-2"),
        ])
    ], className="mb-3")
    
    # Create action buttons
    action_buttons = dbc.Card([
        dbc.CardHeader("Actions"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "View Code", 
                        id={"type": "snapshot-view-button", "index": snapshot_id},
                        color="primary",
                        className="w-100 mb-2"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Restore", 
                        id={"type": "snapshot-restore-button", "index": snapshot_id},
                        color="success",
                        className="w-100 mb-2"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Compare with Parent" if parent_metadata else "No Parent",
                        id={"type": "snapshot-compare-parent-button", "index": snapshot_id},
                        color="info",
                        className="w-100 mb-2",
                        disabled=parent_metadata is None
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Rollback", 
                        id={"type": "snapshot-rollback-button", "index": snapshot_id},
                        color="warning",
                        className="w-100 mb-2"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Tag", 
                        id={"type": "snapshot-tag-button", "index": snapshot_id},
                        color="secondary",
                        className="w-100 mb-2"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Mark as Reverted" if metadata.status != "reverted" else "Already Reverted", 
                        id={"type": "snapshot-revert-button", "index": snapshot_id},
                        color="danger",
                        className="w-100 mb-2",
                        disabled=metadata.status == "reverted"
                    )
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Create relationships card
    relationships_card = dbc.Card([
        dbc.CardHeader("Relationships"),
        dbc.CardBody([
            html.Div([
                html.Strong("Parent: "),
                html.A(
                    parent_metadata.snapshot_id if parent_metadata else "None",
                    id={"type": "snapshot-parent-link", "index": snapshot_id},
                    href="#",
                    className="text-info",
                    style={"textDecoration": "none"}
                ) if parent_metadata else html.Span("None")
            ], className="mb-2"),
            html.Div([
                html.Strong("Children: "),
                html.Div([
                    html.A(
                        s.snapshot_id,
                        id={"type": "snapshot-child-link", "index": s.snapshot_id},
                        href="#",
                        className="text-info mr-2",
                        style={"textDecoration": "none", "marginRight": "10px"}
                    ) for s in next_snapshots
                ]) if next_snapshots else html.Span("None")
            ], className="mb-2")
        ])
    ])
    
    return html.Div([
        info_card,
        action_buttons,
        relationships_card
    ])


def create_tag_input_modal(snapshot_id: str) -> dbc.Modal:
    """
    Create a modal for adding tags to a snapshot.
    
    Args:
        snapshot_id: ID of the snapshot to tag
        
    Returns:
        Modal component for tagging
    """
    return dbc.Modal([
        dbc.ModalHeader("Add Tags to Snapshot"),
        dbc.ModalBody([
            html.P(f"Add tags to snapshot {snapshot_id}:"),
            dbc.Input(
                id="tag-input-field",
                placeholder="Enter tags separated by commas",
                type="text"
            )
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel", 
                id="tag-modal-cancel",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "Add Tags", 
                id={"type": "tag-modal-submit", "index": snapshot_id},
                color="primary"
            )
        ])
    ], id="tag-modal", is_open=False)


def create_rollback_confirm_modal(snapshot_id: str) -> dbc.Modal:
    """
    Create a modal for confirming rollback to a snapshot.
    
    Args:
        snapshot_id: ID of the snapshot to rollback to
        
    Returns:
        Modal component for rollback confirmation
    """
    return dbc.Modal([
        dbc.ModalHeader("Confirm Rollback"),
        dbc.ModalBody([
            html.P(f"Are you sure you want to rollback to snapshot {snapshot_id}?"),
            html.P("This will create a new snapshot with the same content as this one."),
            dbc.Input(
                id="rollback-message-input",
                placeholder="Optional message for the rollback snapshot",
                type="text"
            )
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel", 
                id="rollback-modal-cancel",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "Rollback", 
                id={"type": "rollback-modal-submit", "index": snapshot_id},
                color="warning"
            )
        ])
    ], id="rollback-modal", is_open=False)


def create_snapshot_management_view(engine: SnapshotEngine, file_path: Optional[Union[str, Path]] = None,
                                   branch: Optional[str] = None) -> html.Div:
    """
    Create a comprehensive view for managing snapshots.
    
    Args:
        engine: Snapshot engine instance
        file_path: Optional path to filter snapshots by
        branch: Optional branch to filter snapshots by
        
    Returns:
        Dash component for snapshot management
    """
    # Get snapshots based on filters
    if file_path:
        snapshots = engine.get_snapshots(file_path, branch)
    else:
        snapshots = engine.get_all_snapshots(branch)
    
    # Sort by timestamp (newest first)
    snapshots = sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
    
    # File selector dropdown items
    file_options = []
    all_snapshots = engine.get_all_snapshots()
    unique_files = set(s.file_path for s in all_snapshots)
    
    for f in sorted(unique_files):
        file_options.append({"label": f, "value": f})
    
    # Branch selector dropdown items
    branch_options = [{"label": "All Branches", "value": ""}]
    for branch_name in engine.branches:
        branch_options.append({"label": branch_name, "value": branch_name})
    
    # Create control panel
    control_panel = dbc.Card([
        dbc.CardHeader("Snapshot Filters"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("File:"),
                    dcc.Dropdown(
                        id="snapshot-file-filter",
                        options=[{"label": "All Files", "value": ""}] + file_options,
                        value=file_path or "",
                        clearable=False,
                        style={"backgroundColor": "#3d3d3d", "color": "white"}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Branch:"),
                    dcc.Dropdown(
                        id="snapshot-branch-filter",
                        options=branch_options,
                        value=branch or "",
                        clearable=False,
                        style={"backgroundColor": "#3d3d3d", "color": "white"}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Status:"),
                    dcc.Dropdown(
                        id="snapshot-status-filter",
                        options=[
                            {"label": "All Statuses", "value": ""},
                            {"label": "Success", "value": "success"},
                            {"label": "Failed", "value": "failed"},
                            {"label": "Reverted", "value": "reverted"}
                        ],
                        value="",
                        clearable=False,
                        style={"backgroundColor": "#3d3d3d", "color": "white"}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("", className="mb-2"),  # Empty label for alignment
                    dbc.Button(
                        "Apply Filters",
                        id="snapshot-apply-filters",
                        color="primary",
                        className="w-100"
                    )
                ], width=6, className="d-flex align-items-end")
            ], className="mt-3")
        ])
    ], className="mb-4")
    
    # Create tabs for different views
    tabs = dbc.Tabs([
        dbc.Tab([
            create_snapshot_history_table(snapshots)
        ], label="History", tab_id="history-tab"),
        dbc.Tab([
            create_snapshot_visualization(snapshots)
        ], label="Timeline", tab_id="timeline-tab"),
        dbc.Tab([
            html.Div(id="snapshot-details-container", className="p-3")
        ], label="Details", tab_id="details-tab", disabled=True),
        dbc.Tab([
            html.Div(id="snapshot-diff-container", className="p-3")
        ], label="Diff View", tab_id="diff-tab", disabled=True)
    ], id="snapshot-tabs", active_tab="history-tab")
    
    # Create hidden modals for interactions
    tag_modal = create_tag_input_modal("")  # Will be updated dynamically
    rollback_modal = create_rollback_confirm_modal("")  # Will be updated dynamically
    
    # Create statistics card
    stats = engine.get_snapshot_size_info()
    stats_card = dbc.Card([
        dbc.CardHeader("Snapshot Statistics"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Strong("Total Snapshots: "),
                        html.Span(f"{stats['total_snapshots']:,}")
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Total Size: "),
                        html.Span(f"{stats['total_size_bytes']:,} bytes")
                    ], className="mb-2")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Strong("Average Size: "),
                        html.Span(f"{int(stats['average_size_bytes']):,} bytes")
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Largest Snapshot: "),
                        html.Span(f"{stats['largest_snapshot_size_bytes']:,} bytes")
                    ], className="mb-2")
                ], width=6)
            ])
        ])
    ], className="mb-4")
    
    return html.Div([
        html.H2("Snapshot Management", className="mb-4"),
        control_panel,
        stats_card,
        tabs,
        tag_modal,
        rollback_modal,
        
        # Hidden div to store currently selected snapshot ID
        dcc.Store(id="selected-snapshot-id"),
        
        # Hidden div to store comparison source ID
        dcc.Store(id="comparison-source-id"),
        
        # Hidden div to store comparison target ID
        dcc.Store(id="comparison-target-id")
    ])


def create_code_explorer_with_history(file_path: str, engine: SnapshotEngine) -> html.Div:
    """
    Create a code explorer with snapshot history controls.
    
    Args:
        file_path: Path to the file to display
        engine: Snapshot engine instance
        
    Returns:
        Dash component for code exploring with history
    """
    # Get all snapshots for this file
    snapshots = engine.get_snapshots(file_path)
    snapshots = sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
    
    # Get latest snapshot content
    latest_content = ""
    if snapshots:
        latest_content = engine.get_snapshot_content(snapshots[0].snapshot_id) or ""
    
    # Create snapshot selector
    snapshot_options = []
    for s in snapshots:
        label = f"{s.snapshot_id} - {format_timestamp(s.timestamp)}"
        if s.message:
            label += f" - {s.message}"
        snapshot_options.append({"label": label, "value": s.snapshot_id})
    
    snapshot_selector = html.Div([
        html.Label("Version:"),
        dcc.Dropdown(
            id="code-explorer-snapshot-selector",
            options=snapshot_options,
            value=snapshots[0].snapshot_id if snapshots else None,
            clearable=False,
            style={"backgroundColor": "#3d3d3d", "color": "white"}
        )
    ], className="mb-3")
    
    # Create code display
    code_display = html.Div([
        dbc.Card([
            dbc.CardHeader(f"File: {Path(file_path).name}"),
            dbc.CardBody([
                html.Div(
                    id="code-explorer-content",
                    style={
                        "backgroundColor": "#1e1e1e",
                        "fontFamily": "monospace",
                        "padding": "10px",
                        "borderRadius": "4px",
                        "whiteSpace": "pre-wrap",
                        "overflowX": "auto",
                        "minHeight": "400px"
                    },
                    children=html.Pre(latest_content)
                )
            ])
        ])
    ])
    
    # Create history navigation controls
    history_controls = html.Div([
        dbc.ButtonGroup([
            dbc.Button(
                "Previous Version",
                id="code-explorer-prev-button",
                color="secondary",
                disabled=len(snapshots) <= 1,
                className="mr-2"
            ),
            dbc.Button(
                "Next Version",
                id="code-explorer-next-button",
                color="secondary",
                disabled=len(snapshots) <= 1,
                className="mr-2"
            ),
            dbc.Button(
                "Latest Version",
                id="code-explorer-latest-button",
                color="primary",
                disabled=len(snapshots) <= 1
            )
        ], className="mb-3")
    ])
    
    # Create comparison controls
    comparison_controls = html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Compare with:"),
                dcc.Dropdown(
                    id="code-explorer-compare-selector",
                    options=[{"label": "None", "value": ""}] + snapshot_options,
                    value="",
                    clearable=False,
                    style={"backgroundColor": "#3d3d3d", "color": "white"}
                )
            ], width=8),
            dbc.Col([
                html.Label(" "),  # Empty label for alignment
                dbc.Button(
                    "Compare",
                    id="code-explorer-compare-button",
                    color="info",
                    className="w-100",
                    style={"marginTop": "10px"}
                )
            ], width=4)
        ])
    ], className="mb-3")
    
    # Create restore button
    restore_button = html.Div([
        dbc.Button(
            "Restore This Version",
            id="code-explorer-restore-button",
            color="success",
            className="mb-3"
        )
    ])
    
    return html.Div([
        html.H2(f"Code Explorer: {Path(file_path).name}", className="mb-4"),
        snapshot_selector,
        history_controls,
        comparison_controls,
        restore_button,
        code_display,
        
        # Hidden div for comparison result
        html.Div(id="code-explorer-comparison-result", className="mt-4"),
        
        # Hidden stores
        dcc.Store(id="code-explorer-file-path", data=file_path),
        dcc.Store(id="code-explorer-snapshot-list", data=[s.snapshot_id for s in snapshots])
    ])