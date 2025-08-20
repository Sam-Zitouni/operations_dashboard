import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import plotly.express as px
import time
import streamlit as st
st.write(st.secrets)

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Operations Manager Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# AUTO-REFRESH (every 60 sec)
# ==============================
REFRESH_INTERVAL = 60  # seconds

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()


# If 60s have passed, rerun the app
if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()


# ==============================
# DB CONNECTION
# ==============================
DB_DIALECT = st.secrets["DB_DIALECT"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

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
st.caption(f"Timezone: Africa/Tunis | Auto-refresh every {REFRESH_INTERVAL}s")

# ==============================
# LOAD DATA
# ==============================
trip_df = get_df("""
    SELECT td.trip_id, td.start, td.end, td.real_duration, td.expected_duration,
           bt.type AS stop_type, bt.time AS stop_time
    FROM trip_durations td
    LEFT JOIN bus_trackings bt ON td.trip_id = bt.trip
    ORDER BY td.start DESC
    LIMIT 100
""")

vehicle_df = get_df("SELECT * FROM vehicles")
maintenance_df = get_df("""
    SELECT sm.id, sm.date, sm.vehicle_id, sm.bus_status, sm.manager_confirmation
    FROM scheduled_maintenances sm
""")
vehicle_events_df = get_df("SELECT * FROM vehicle_events")

users_df = get_df("SELECT id, firstname, lastname, position FROM users")
attendance_df = get_df("SELECT * FROM attendances")

# ==============================
# KPIs
# ==============================
st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

# 1️⃣ On-Time Departure & Arrival %
if not trip_df.empty:
    on_time_pct = round(
        (trip_df['real_duration'] <= trip_df['expected_duration']).mean() * 100, 2
    )
else:
    on_time_pct = 0
col1.metric("On-Time Departure/Arrival %", f"{on_time_pct}%")

# 2️⃣ Trip Completion & Cancellation
if not trip_df.empty:
    trip_total = len(trip_df)
    trip_completed = (trip_df['real_duration'].notnull()).sum()
    trip_cancelled = trip_total - trip_completed
else:
    trip_total, trip_completed, trip_cancelled = 0, 0, 0
col2.metric("Trip Completion", trip_completed)
col2.metric("Trip Cancelled", trip_cancelled)

# 3️⃣ Fleet Downtime & MTBF
fleet_downtime = vehicle_events_df.shape[0] if not vehicle_events_df.empty else 0
col3.metric("Fleet Downtime Events", fleet_downtime)
mtbf = 120  # placeholder in hours
col3.metric("Mean Time Between Failures (MTBF)", f"{mtbf} hrs")

# 4️⃣ Staff Punctuality / Readiness
if not attendance_df.empty and "presence_type" in attendance_df.columns:
    staff_present_pct = round((attendance_df['presence_type'] == 'present').mean() * 100, 2)
else:
    staff_present_pct = 0
col4.metric("Staff Readiness %", f"{staff_present_pct}%")

# ==============================
# VISUALS
# ==============================
st.subheader("Visualizations")

# Delay Root Cause Pie Chart
st.markdown("**Delay Root Cause Pie Chart**")
if not trip_df.empty:
    trip_df['delay'] = trip_df['real_duration'] - trip_df['expected_duration']
    delay_summary = trip_df.groupby('stop_type', dropna=True)['delay'].sum().reset_index()
    if not delay_summary.empty:
        fig1 = px.pie(delay_summary, names='stop_type', values='delay', title="Delay by Stop Type")
        st.plotly_chart(fig1, use_container_width=True)

# Vehicle Reliability Scorecard
st.markdown("**Vehicle Reliability Scorecard**")
if not vehicle_df.empty and not vehicle_events_df.empty:
    events_per_vehicle = vehicle_events_df.groupby("vehicle_id")["id"].count().reset_index()
    events_per_vehicle.rename(columns={"id": "events"}, inplace=True)
    vehicle_reliability = vehicle_df.merge(
        events_per_vehicle, left_on="id", right_on="vehicle_id", how="left"
    )
    vehicle_reliability["events"] = vehicle_reliability["events"].fillna(0)
    fig2 = px.bar(vehicle_reliability, x='registration_number', y='events',
                  title="Vehicle Events Count")
    st.plotly_chart(fig2, use_container_width=True)

# Crew Performance Radar Chart (sample data)
st.markdown("**Crew Performance by Route (Radar Chart)**")
crew_perf = pd.DataFrame({
    'driver': ['Driver A', 'Driver B', 'Driver C'],
    'route1': [90, 80, 85],
    'route2': [88, 92, 79],
    'route3': [95, 85, 90]
})
crew_perf_long = crew_perf.melt(id_vars="driver", var_name="route", value_name="score")
fig3 = px.line_polar(crew_perf_long, r="score", theta="route", color="driver",
                     line_close=True, markers=True, title="Crew Performance Radar")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# QUICK ACTIONS
# ==============================
st.subheader("Quick Actions / Shortcuts")

with st.expander("🚦 Flag delayed trips"):
    if not trip_df.empty:
        delayed_trips = trip_df[trip_df['real_duration'] > trip_df['expected_duration']]
        st.dataframe(delayed_trips[['trip_id','start','end','real_duration','expected_duration']])
    else:
        st.write("No trip data available.")

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
st.caption("Dashboard powered by Streamlit | Last updated: " +
           datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

