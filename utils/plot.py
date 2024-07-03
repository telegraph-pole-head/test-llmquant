import plotly.graph_objects as go
from datetime import datetime

def show_timeline(data):
    # Convert timestamps to human-readable format
    for key in data:
        if 'time' in key and key != 'time_in_queue':
            data[key] = datetime.fromtimestamp(data[key])

    # Create a timeline plot
    fig = go.Figure()

    # Adding points and lines for each event
    events = ['arrival_time', 'last_token_time', 'first_scheduled_time', 'first_token_time', 'finished_time']
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i, event in enumerate(events):
        fig.add_trace(go.Scatter(
            x=[data[event], data[event]],
            y=[0, 1],
            mode="lines+markers",
            name=event,
            text=event,
            textposition="top right",
            marker=dict(color=colors[i])
        ))

    # Update layout
    fig.update_layout(
        title="Latency Visualization",
        xaxis_title="Time",
        yaxis_title="Event",
        yaxis=dict(showticklabels=False),
        showlegend=False
    )

    fig.show()
