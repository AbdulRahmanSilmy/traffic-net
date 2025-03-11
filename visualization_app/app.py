import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from st_files_connection import FilesConnection

st.set_page_config(page_title="Traffic Analysis", layout="wide")


st_autorefresh(interval=86400 * 1000, key="data_refresh")
# Load data

conn = st.connection('gcs', type=FilesConnection)
df_plot = conn.read("traffic_net/traffic_147_two_way.csv",
                    input_format="csv", ttl=600)
df_plot.set_index('time', inplace=True)
df_plot.index = pd.to_datetime(df_plot.index)

# Create the figure
fig = go.Figure()

# Add trace for real data
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['num_cars'],
    mode="lines+markers",
    name="Real Data",
    opacity=0.5,
    # marker=dict(color="lightblue")
    marker=dict(color="#6699CC")
))

# Add trace for forecast data
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['forecast'],
    mode="lines+markers",
    name="Forecast Data",
    opacity=0.5,
    marker=dict(color="purple")  # Changed color to purple
))

# Calculate the range for the last 7 days
end_date = df_plot.index.max() + timedelta(hours=3)
start_date = end_date - timedelta(days=7)

# Update layout
fig.update_layout(
    title="Traffic Analysis of the William R. Bennett Bridge",
    xaxis=dict(
        title="Date",
        type="date",  # Ensures proper date handling
        rangeslider=dict(visible=True),  # Enables scrolling
        # Set the initial range to the last 7 days
        range=[start_date, end_date],
        scaleanchor="x",  # Set the scale anchor to x
        scaleratio=8/4
    ),
    yaxis=dict(
        title="Average number of cars",
        scaleanchor="y",  # Set the scale anchor to y
    ),
    annotations=[dict(
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        showarrow=False,
        text="For more details, visit <a href='https://github.com/AbdulRahmanSilmy/traffic-net'>GitHub</a>"
    )]
)

# Display the figure using Streamlit
st.plotly_chart(fig)
