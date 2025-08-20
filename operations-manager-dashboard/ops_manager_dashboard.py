import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Operations Manager Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 60 sec
refresh_interval_ms = 60000  # 1 min
count = st_autorefresh(interval=refresh_interval_ms, key="datarefresh")

# ==============================
# DB CONNECTION
# ==============================
DB_DIALECT = st.secrets.get("DB_DIALECT", "postgresql")
DB_HOST = st.secrets.get("DB_HOST")
DB_PORT = st.secrets.get("DB_PORT")
DB_NAME = st.secrets.get("DB_NAME")
DB_USER = st.secrets.get("DB_USER")
DB_PASSWORD = st.secrets.get("DB_PASSWORD")

engine = create_engine(
    f"{DB_DIALECT}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ==============================
# HELPER FUNCTIONS
# ==============================
@st.cache_data(ttl=60)  # cache for 60 sec
def get_df(query):
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"DB Query Error: {e}")
        return pd.DataFrame()

# ==============================
# PAGE LAYOUT
# ==============================
st.title("🚍 Operations Manager Dashboard")
st.caption(f"Timezone: Africa/Tunis | Auto-refresh every {refresh_interval_ms/1000}s")

# ==============================
# LOAD DATA
# ==============================
# Trips and timings
trip_df = get_df("""
    SELECT td.trip_id, td.start, td.end, td.real_duration, td.expected_duration,
           bt.type AS stop_type, bt.time AS stop_time
    FROM trip_durations td
    LEFT JOIN bus_trackings bt ON td.trip_id = bt.trip
    ORDER BY td.start DESC
    LIMIT 100
""")

# Vehicles & Maintenance
vehicle_df = get_df("SELECT * FROM vehicles")
maintenance_df = get_df("""
    SELECT sm.id, sm.date, sm.vehicle_id, sm.bus_status, sm.manager_confirmation
    FROM scheduled_maintenances sm
""")
vehicle_events_df = get_df("SELECT * FROM vehicle_events")

# Staff & Attendance
users_df = get_df("SELECT id, firstname, lastname, position FROM users")
attendance_df = get_df("SELECT * FROM attendances")

# ==============================
# KPIs
# ==============================
st.subheader("Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)

# 1️⃣ On-Time Departure & Arrival %
on_time_pct = round(
    (trip_df['real_duration'] <= trip_df['expected_duration']).mean()*100, 2
)
col1.metric("On-Time Departure/Arrival %", f"{on_time_pct}%")

# 2️⃣ Trip Completion & Cancellation
trip_total = len(trip_df)
trip_completed = (trip_df['real_duration'].notnull()).sum()
trip_cancelled = trip_total - trip_completed
col2.metric("Trip Completion", trip_completed)
col2.metric("Trip Cancelled", trip_cancelled)

# 3️⃣ Fleet Downtime & MTBF
fleet_downtime = vehicle_events_df.shape[0]  # placeholder
col3.metric("Fleet Downtime Events", fleet_downtime)
mtbf = 120  # placeholder in hours
col3.metric("Mean Time Between Failures (MTBF)", f"{mtbf} hrs")

# 4️⃣ Staff Punctuality / Readiness
staff_present_pct = round((attendance_df['presence_type'] == 'present').mean()*100,2)
col4.metric("Staff Readiness %", f"{staff_present_pct}%")

# ==============================
# VISUALS
# ==============================
st.subheader("Visualizations")

# Delay Root Cause Pie Chart
st.markdown("**Delay Root Cause Pie Chart**")
if not trip_df.empty:
    trip_df['delay'] = trip_df['real_duration'] - trip_df['expected_duration']
    delay_summary = trip_df.groupby('stop_type')['delay'].sum().reset_index()
    fig1 = px.pie(delay_summary, names='stop_type', values='delay', title="Delay by Stop Type")
    st.plotly_chart(fig1, use_container_width=True)

# Vehicle Reliability Scorecard
st.markdown("**Vehicle Reliability Scorecard**")
vehicle_reliability = vehicle_df.copy()
vehicle_reliability['events'] = vehicle_events_df.groupby('vehicle_id')['id'].transform('count')
fig2 = px.bar(vehicle_reliability, x='registration_number', y='events', title="Vehicle Events Count")
st.plotly_chart(fig2, use_container_width=True)

# Crew Performance Radar Chart (placeholder)
st.markdown("**Crew Performance by Route (Radar Chart)**")
crew_perf = pd.DataFrame({
    'driver': ['Driver A', 'Driver B', 'Driver C'],
    'route1': [90, 80, 85],
    'route2': [88, 92, 79],
    'route3': [95, 85, 90]
})
fig3 = px.line_polar(crew_perf, r=['route1','route2','route3'], theta=['route1','route2','route3'],
                     line_close=True, title="Crew Performance Radar", markers=True)
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# QUICK ACTIONS
# ==============================
st.subheader("Quick Actions / Shortcuts")

with st.expander("🚦 Flag delayed trips"):
    delayed_trips = trip_df[trip_df['real_duration'] > trip_df['expected_duration']]
    st.dataframe(delayed_trips[['trip_id','start','end','real_duration','expected_duration']])

with st.expander("🧑 Assign new driver/steward"):
    st.write("⚡ This feature requires integration with your users & trip assignment logic")

with st.expander("🛠 Request bus maintenance"):
    st.write("⚡ This would create a new record in `scheduled_maintenances`")

with st.expander("🗓 Approve trip confirmation"):
    st.write("⚡ Approve trips that are pending confirmation")

# ==============================
# END OF DASHBOARD
# ==============================
st.markdown("---")
st.caption("Dashboard powered by Streamlit | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
