import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load some sample data for demonstration
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
    'Amount': [4, 1, 2, 2, 4, 5],
    'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
})

# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the CSS for dark theme with blue and purple gradients
app.css.append_css({
    "external_url": [
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
    ]
})

# Define layout
app.layout = html.Div(
    style={
        'backgroundColor': '#121212',
        'color': '#FFFFFF'
    },
    children=[
        # Header
        html.Header(
            className="header",
            style={
                'background': "linear-gradient(90deg, #4A90E2 0%, #8E54E9 100%)",
                'padding': '20px',
                'text-align': 'center'
            },
            children=[
                html.H1("Dash App with Dark Theme and Gradients", style={'color': '#FFFFFF'})
            ]
        ),
        
        # Main content
        html.Div(
            className="content",
            style={
                'padding': '20px',
                'background': "linear-gradient(90deg, #4A90E2 0%, #8E54E9 100%)"
            },
            children=[
                dcc.Graph(id='example-graph', figure=px.bar(df, x='Fruit', y='Amount', color='City')),
                
                html.Div(
                    style={
                        'color': '#FFFFFF',
                        'fontFamily': 'Arial, sans-serif'
                    },
                    children=[
                        "This is a simple example of a Dash app in dark mode with blue and purple gradients."
                    ]
                ),
                
                dcc.Graph(id='map-graph', figure={})
            ]
        ),
        
        # Footer
        html.Footer(
            className="footer",
            style={
                'background': "linear-gradient(90deg, #4A90E2 0%, #8E54E9 100%)",
                'padding': '20px',
                'text-align': 'center'
            },
            children=[
                html.P("Created by Alibaba Cloud", style={'color': '#FFFFFF'})
            ]
        )
    ]
)

# Callback to update the map
@app.callback(
    Output('map-graph', 'figure'),
    Input('example-graph', 'clickData')
)
def update_map(click_data):
    if click_data is not None:
        city = click_data['points'][0]['label']
        df_city = df[df['City'] == city]
        
        fig = px.scatter_geo(df_city, locations='City', location_mode='country names',
                             hover_name='Fruit', size='Amount', color_discrete_sequence=['#4A90E2'])
        return fig
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)
