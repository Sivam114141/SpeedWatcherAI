import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Set page config
st.set_page_config(page_title="Vehicle Speed Dashboard", layout="wide")

# Load data
LOG_PATH = "vehicle_log.csv"
if os.path.exists(LOG_PATH):
    df = pd.read_csv(LOG_PATH)
else:
    st.error("Log file not found.")
    st.stop()

# Convert Timestamp to datetime if it's not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sidebar filters
st.sidebar.header("Filters")
location_filter = st.sidebar.selectbox("Select Location", options=['All'] + sorted(df['Location'].dropna().unique().tolist()))
status_filter = st.sidebar.selectbox("Select Status", options=['All', 'OK', 'Overspeed'])

# Apply filters
filtered_df = df.copy()
if location_filter != 'All':
    filtered_df = filtered_df[filtered_df['Location'] == location_filter]
if status_filter != 'All':
    filtered_df = filtered_df[filtered_df['Status'] == status_filter]

# Title
st.title("Vehicle Speed Monitoring Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Vehicles", len(filtered_df))
with col2:
    st.metric("Overspeeding", filtered_df[filtered_df['Status'] == 'Overspeed'].shape[0])
with col3:
    avg_speed = filtered_df['Speed'].mean() if not filtered_df.empty else 0
    st.metric("Average Speed", f"{avg_speed:.1f} km/h")

# Speed distribution chart
st.subheader("Speed Distribution")
speed_chart = px.histogram(filtered_df, x='Speed', nbins=20, color='Status', title="Vehicle Speed Distribution")
st.plotly_chart(speed_chart, use_container_width=True)

# Table
st.subheader("Detailed Log")
st.dataframe(filtered_df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
