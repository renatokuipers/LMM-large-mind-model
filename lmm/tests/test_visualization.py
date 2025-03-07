import pytest
import os
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from lmm.visualization import (
    dashboard,
    metrics
)

class TestDashboard:
    """Tests for the dashboard visualization functionality."""
    
    @pytest.fixture
    def mock_app(self):
        """Fixture providing a mock Dash app."""
        mock = MagicMock()
        mock.layout = None
        return mock
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for dashboard visualization."""
        return {
            "time_series": pd.DataFrame({
                "timestamp": pd.date_range(start="2023-01-01", periods=10),
                "value": np.random.randn(10),
                "category": ["A", "B"] * 5
            }),
            "categorical": pd.DataFrame({
                "category": ["A", "B", "C", "D"],
                "count": [10, 15, 7, 12]
            })
        }
    
    def test_dashboard_initialization(self, mock_app):
        """Test dashboard initialization."""
        with patch("dash.Dash", return_value=mock_app):
            # Initialize dashboard with default settings
            dash_instance = dashboard.create_dashboard()
            
            # Verify initialization
            assert dash_instance is not None
            assert dash_instance == mock_app
            assert mock_app.layout is not None
    
    def test_dashboard_components(self, mock_app, sample_data):
        """Test dashboard components creation."""
        with patch("dash.Dash", return_value=mock_app), \
             patch("lmm.visualization.dashboard.create_time_series_graph") as mock_time_series, \
             patch("lmm.visualization.dashboard.create_categorical_graph") as mock_categorical:
            
            # Configure mocks to return components
            mock_time_series.return_value = MagicMock()
            mock_categorical.return_value = MagicMock()
            
            # Initialize dashboard with sample data
            dash_instance = dashboard.create_dashboard(data=sample_data)
            
            # Verify that graph creation functions were called
            mock_time_series.assert_called_once()
            mock_categorical.assert_called_once()
            
            # Check that the dashboard layout was set
            assert mock_app.layout is not None
    
    @pytest.mark.parametrize("data_type,expected_components", [
        ("time_series", ["line-chart", "date-picker"]),
        ("categorical", ["bar-chart", "dropdown"])
    ])
    def test_dashboard_layout_by_data_type(self, mock_app, sample_data, data_type, expected_components):
        """Test dashboard layout changes based on data type."""
        with patch("dash.Dash", return_value=mock_app), \
             patch("lmm.visualization.dashboard.create_time_series_graph") as mock_time_series, \
             patch("lmm.visualization.dashboard.create_categorical_graph") as mock_categorical, \
             patch("dash.html.Div") as mock_div, \
             patch("dash.html.H1") as mock_h1, \
             patch("dash.html.P") as mock_p, \
             patch("dash.dcc.DatePickerRange") as mock_date_picker, \
             patch("dash.dcc.Dropdown") as mock_dropdown:
            
            # Configure mocks to return components
            mock_time_series.return_value = MagicMock(id="line-chart")
            mock_categorical.return_value = MagicMock(id="bar-chart")
            mock_date_picker.return_value = MagicMock(id="date-picker")
            mock_dropdown.return_value = MagicMock(id="dropdown")
            
            # Initialize dashboard with specific data type focus
            dash_instance = dashboard.create_dashboard(
                data=sample_data,
                focus_data_type=data_type
            )
            
            # Verify components based on data type
            if data_type == "time_series":
                mock_time_series.assert_called_once()
                mock_date_picker.assert_called_once()
            elif data_type == "categorical":
                mock_categorical.assert_called_once()
                mock_dropdown.assert_called_once()

class TestMetrics:
    """Tests for the metrics visualization functionality."""
    
    @pytest.fixture
    def performance_data(self):
        """Fixture providing performance metrics data."""
        return {
            "accuracy": [0.8, 0.82, 0.85, 0.9],
            "precision": [0.75, 0.78, 0.8, 0.85],
            "recall": [0.7, 0.72, 0.75, 0.8],
            "f1": [0.72, 0.75, 0.77, 0.82],
            "timestamp": pd.date_range(start="2023-01-01", periods=4)
        }
    
    @pytest.fixture
    def embedding_data(self):
        """Fixture providing embedding data for visualization."""
        # Create random 2D embeddings for visualization
        embedding_vectors = np.random.randn(20, 2)
        labels = ["A"] * 10 + ["B"] * 10
        return pd.DataFrame({
            "x": embedding_vectors[:, 0],
            "y": embedding_vectors[:, 1],
            "label": labels
        })
    
    def test_performance_plot(self, performance_data):
        """Test creation of performance metrics plots."""
        with patch("plotly.graph_objects.Figure") as mock_figure, \
             patch("plotly.subplots.make_subplots") as mock_make_subplots:
            
            # Configure mocks
            mock_subplot = MagicMock()
            mock_make_subplots.return_value = mock_subplot
            mock_subplot.add_trace = MagicMock()
            
            # Create performance plot
            fig = metrics.plot_performance_metrics(performance_data)
            
            # Verify subplots creation
            mock_make_subplots.assert_called_once()
            
            # Verify traces were added
            assert mock_subplot.add_trace.call_count >= 4  # One trace for each metric
            
            # Verify figure customization
            mock_subplot.update_layout.assert_called_once()
    
    def test_embedding_visualization(self, embedding_data):
        """Test visualization of embeddings."""
        with patch("plotly.graph_objects.Figure") as mock_figure, \
             patch("plotly.graph_objects.Scatter") as mock_scatter:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_figure.return_value = mock_fig_instance
            
            # Create embedding visualization
            fig = metrics.plot_embeddings(
                x=embedding_data["x"],
                y=embedding_data["y"],
                labels=embedding_data["label"]
            )
            
            # Verify figure creation
            mock_figure.assert_called_once()
            
            # Verify scatter plot creation
            assert mock_scatter.call_count >= 1
            
            # Verify figure customization
            mock_fig_instance.update_layout.assert_called_once()
    
    @pytest.mark.parametrize("metric_name,expected_trend", [
        ("accuracy", "increasing"),
        ("precision", "increasing"),
        ("recall", "increasing"),
        ("f1", "increasing")
    ])
    def test_metric_trends(self, performance_data, metric_name, expected_trend):
        """Test calculation of metric trends."""
        # Calculate trend
        trend = metrics.calculate_trend(performance_data[metric_name])
        
        # Verify trend calculation
        if expected_trend == "increasing":
            assert trend > 0  # Positive slope indicates increasing trend
        elif expected_trend == "decreasing":
            assert trend < 0  # Negative slope indicates decreasing trend
        else:
            assert abs(trend) < 0.01  # Near-zero slope indicates stable trend
    
    def test_confusion_matrix_plot(self):
        """Test creation of confusion matrix plot."""
        # Create sample confusion matrix
        conf_matrix = np.array([
            [45, 5],
            [10, 40]
        ])
        
        with patch("plotly.graph_objects.Figure") as mock_figure, \
             patch("plotly.graph_objects.Heatmap") as mock_heatmap:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_figure.return_value = mock_fig_instance
            
            # Create confusion matrix plot
            fig = metrics.plot_confusion_matrix(
                conf_matrix=conf_matrix,
                class_names=["Negative", "Positive"]
            )
            
            # Verify figure creation
            mock_figure.assert_called_once()
            
            # Verify heatmap creation
            mock_heatmap.assert_called_once()
            
            # Verify figure customization
            mock_fig_instance.update_layout.assert_called_once()
    
    def test_create_gauge_chart(self):
        """Test creation of gauge chart."""
        with patch("plotly.graph_objects.Figure") as mock_figure, \
             patch("plotly.graph_objects.Indicator") as mock_indicator:
            
            # Configure mocks
            mock_fig_instance = MagicMock()
            mock_figure.return_value = mock_fig_instance
            
            # Create gauge chart
            fig = metrics.create_gauge_chart(
                value=75,
                title="Test Gauge",
                min_value=0,
                max_value=100
            )
            
            # Verify figure creation
            mock_figure.assert_called_once()
            
            # Verify indicator creation
            mock_indicator.assert_called_once()
            
            # Verify figure customization
            mock_fig_instance.update_layout.assert_called_once() 