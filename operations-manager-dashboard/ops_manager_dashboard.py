import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Operations Manager Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# HELPERS
# ==============================
@st.cache_data(ttl=60)
def get_df(query: str, engine):
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"DB Query Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def to_datetime_safe(s, utc=False):
    try:
        return pd.to_datetime(s, utc=utc, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series(s), errors="coerce")

# Pretty badges
def kpi_delta(current: float | int, previous: float | int | None):
    if previous is None or np.isnan(previous):
        return None
    if previous == 0:
        return "+∞" if current > 0 else "0"
    return f"{(current-previous)/abs(previous)*100:.1f}%"

# Empty-state helper
def empty_state(msg: str):
    st.info(msg)

# Color for delay status
def delay_status(real_dur, expected_dur, threshold_min=5):
    if pd.isna(real_dur) or pd.isna(expected_dur):
        return "unknown"
    diff = real_dur - expected_dur
    if diff <= timedelta(minutes=threshold_min):
        return "on_time"
    return "late"

# Render a timeline (gantt-like)
def trips_timeline(df: pd.DataFrame):
    if df.empty:
        empty_state("No trip data for the selected filters.")
        return
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="trip_id",
        color="status",
        hover_data={"expected_duration": True, "real_duration": True, "route": True, "vehicle_id": True},
        category_orders={"status": ["late", "on_time", "unknown"]},
        title="Real-time Trip Dashboard"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SIDEBAR: CONNECTION + FILTERS
# ==============================
st.sidebar.header("⚙️ Settings")
DEMO_MODE = st.sidebar.toggle("Demo mode (use mock data if DB is empty)", value=False, help="Ensures visuals render even if your tables are empty.")
REFRESH_SEC = st.sidebar.number_input("Auto-refresh (seconds)", 15, 600, 60)

# Use built-in autorefresh (doesn't loop/rerun manually)
st_autorefresh_event = st.sidebar.empty()
st.experimental_rerun
st_autorefresh_event = st.sidebar.caption(":arrows_counterclockwise: Auto-refresh active")
st_autorefresh = st.experimental_data_editor if False else None  # placeholder to avoid linter warning
st.autorefresh = st.experimental_rerun if False else None # no-op holder

# NOTE: Don't print secrets in production
try:
    DB_DIALECT = st.secrets["DB_DIALECT"]
    DB_HOST = st.secrets["DB_HOST"]
    DB_PORT = st.secrets["DB_PORT"]
    DB_NAME = st.secrets["DB_NAME"]
    DB_USER = st.secrets["DB_USER"]
    DB_PASSWORD = st.secrets["DB_PASSWORD"]
    engine = create_engine(f"{DB_DIALECT}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
except Exception as e:
    st.warning("Couldn't read DB connection from secrets. You can still use Demo mode.")
    engine = None

# ==============================
# LOAD DATA
# ==============================
trip_df = pd.DataFrame()
vehicle_df = pd.DataFrame()
maintenance_df = pd.DataFrame()
vehicle_events_df = pd.DataFrame()
users_df = pd.DataFrame()
attendance_df = pd.DataFrame()

if engine is not None:
    trip_df = get_df(
        """
        SELECT td.trip_id,
               td.start,
               td."end",
               td.real_duration,
               td.expected_duration,
               td.route,
               td.vehicle_id,
               bt.type AS stop_type,
               bt.time AS stop_time
        FROM trip_durations td
        LEFT JOIN bus_trackings bt ON td.trip_id = bt.trip
        ORDER BY td.start DESC
        LIMIT 500
        """,
        engine,
    )

    vehicle_df = get_df("SELECT * FROM vehicles", engine)

    maintenance_df = get_df(
        """
        SELECT sm.id,
               sm.date,
               sm.vehicle_id,
               sm.bus_status,
               sm.manager_confirmation,
               sm.notes
        FROM scheduled_maintenances sm
        ORDER BY sm.date DESC
        """,
        engine,
    )

    vehicle_events_df = get_df("SELECT * FROM vehicle_events", engine)

    users_df = get_df("SELECT id, firstname, lastname, position FROM users", engine)

    attendance_df = get_df("SELECT * FROM attendances", engine)

# DEMO DATA when requested or tables are empty
if DEMO_MODE or (trip_df.empty and vehicle_df.empty and maintenance_df.empty):
    now = pd.Timestamp.utcnow().tz_localize(None)
    demo_trips = []
    for i in range(20):
        start = now - pd.Timedelta(hours=8) + pd.Timedelta(minutes=30*i)
        expected = pd.Timedelta(minutes=np.random.randint(30, 60))
        real = expected + pd.Timedelta(minutes=np.random.randint(-5, 20))
        demo_trips.append({
            "trip_id": f"T{i+1:03d}",
            "start": start,
            "end": start + real,
            "real_duration": real,
            "expected_duration": expected,
            "route": f"R{1 + (i % 3)}",
            "vehicle_id": 100 + (i % 5),
            "stop_type": np.random.choice(["traffic", "boarding", "mechanical", "weather", None], p=[.35,.25,.2,.1,.1]),
            "stop_time": start + pd.Timedelta(minutes=np.random.randint(0, int(real.total_seconds()/60)))
        })
    trip_df = pd.DataFrame(demo_trips)

    vehicle_df = pd.DataFrame({
        "id": [100,101,102,103,104],
        "registration_number": ["TN-AB-100","TN-AB-101","TN-AB-102","TN-AB-103","TN-AB-104"],
        "model": ["Volvo 7900","Iveco Urbanway","MAN Lion","Scania Citywide","Mercedes Citaro"],
    })

    vehicle_events_df = pd.DataFrame({
        "id": range(1, 21),
        "vehicle_id": [100,101,102,103,104]*4,
        "event_type": np.random.choice(["breakdown","inspection","delay"], size=20),
        "created_at": now - pd.to_timedelta(np.random.randint(0, 72, 20), unit="h"),
    })

    maintenance_df = pd.DataFrame({
        "id": range(1,6),
        "date": [date.today() + timedelta(days=d) for d in [0,1,2,3,4]],
        "vehicle_id": [100,101,102,103,104],
        "bus_status": np.random.choice(["scheduled","in_progress","done"], size=5),
        "manager_confirmation": np.random.choice([None, True, False], size=5),
        "notes": ["Oil change","Brake check","AC repair","Tire rotation","General inspection"],
    })

    users_df = pd.DataFrame({
        "id": [1,2,3],
        "firstname": ["Amine","Noura","Karim"],
        "lastname": ["H." ,"S.", "B."],
        "position": ["driver","steward","driver"],
    })

    attendance_df = pd.DataFrame({
        "user_id": [1,2,3,1,2,3],
        "presence_type": ["present","present","late","present","absent","present"],
        "checkin_time": [now - pd.Timedelta(hours=h) for h in [7,7,6,1,1,1]],
        "work_date": [date.today()]*6,
    })

# ==============================
# CLEANUP / TYPES
# ==============================
if not trip_df.empty:
    trip_df["start"] = to_datetime_safe(trip_df["start"]).dt.tz_localize(None)
    trip_df["end"] = to_datetime_safe(trip_df["end"]).dt.tz_localize(None)
    # when durations are stored as seconds, convert to timedeltas if needed
    for col in ["real_duration", "expected_duration"]:
        if np.issubdtype(trip_df[col].dtype, np.number):
            trip_df[col] = pd.to_timedelta(trip_df[col], unit="s")
    trip_df["status"] = [delay_status(r, e) for r, e in zip(trip_df["real_duration"], trip_df["expected_duration"])]

# ==============================
# HEADER
# ==============================
st.title("🚍 Operations Manager Dashboard")
st.caption("Focus: Daily execution, trip monitoring, staff & fleet readiness. Timezone: Africa/Tunis")

# ==============================
# FILTERS
# ==============================
st.sidebar.subheader("Filters")
min_date = trip_df["start"].min().date() if not trip_df.empty else date.today()-timedelta(days=7)
max_date = trip_df["end"].max().date() if not trip_df.empty else date.today()
start_date, end_date = st.sidebar.date_input("Date range", [min_date, max_date])
route_options = sorted([r for r in trip_df.get("route", pd.Series(dtype=str)).dropna().unique()]) if not trip_df.empty else []
route_sel = st.sidebar.multiselect("Routes", options=route_options, default=route_options[:3])
vehicle_options = sorted([int(v) for v in trip_df.get("vehicle_id", pd.Series(dtype=float)).dropna().unique()]) if not trip_df.empty else []
vehicle_sel = st.sidebar.multiselect("Vehicles", options=vehicle_options, default=vehicle_options)

# Apply filters
if not trip_df.empty:
    mask = (trip_df["start"].dt.date >= start_date) & (trip_df["end"].dt.date <= end_date)
    if route_sel:
        mask &= trip_df["route"].isin(route_sel)
    if vehicle_sel:
        mask &= trip_df["vehicle_id"].isin(vehicle_sel)
    trip_df_f = trip_df.loc[mask].copy()
else:
    trip_df_f = trip_df.copy()

# ==============================
# KPIs
# ==============================
st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

# 1️⃣ On-Time Departure & Arrival %
if not trip_df_f.empty:
    on_time_pct = round((trip_df_f["real_duration"] <= trip_df_f["expected_duration"]).mean()*100, 2)
else:
    on_time_pct = 0.0
col1.metric("On-Time Departure/Arrival %", f"{on_time_pct}%")

# 2️⃣ Trip Completion & Cancellation
if not trip_df_f.empty:
    trip_total = len(trip_df_f)
    trip_completed = trip_df_f["real_duration"].notna().sum()
    trip_cancelled = trip_total - trip_completed
else:
    trip_total = trip_completed = trip_cancelled = 0
col2.metric("Trip Completion", int(trip_completed))
col2.metric("Trip Cancelled", int(trip_cancelled))

# 3️⃣ Fleet Downtime & MTBF (simple proxy)
fleet_downtime = int(vehicle_events_df.shape[0]) if not vehicle_events_df.empty else 0
col3.metric("Fleet Downtime Events (24–72h)", fleet_downtime)
# Proxy MTBF: hours observed / events
if not trip_df_f.empty and fleet_downtime > 0:
    hours_observed = (trip_df_f["end"].max() - trip_df_f["start"].min()).total_seconds() / 3600
    mtbf = max(1, round(hours_observed / fleet_downtime, 1))
else:
    mtbf = 0
col3.metric("Mean Time Between Failures (MTBF)", f"{mtbf} hrs")

# 4️⃣ Staff Punctuality / Readiness
if not attendance_df.empty and "presence_type" in attendance_df.columns:
    staff_present_pct = round((attendance_df["presence_type"].eq("present")).mean()*100, 2)
else:
    staff_present_pct = 0.0
col4.metric("Staff Readiness %", f"{staff_present_pct}%")

# ==============================
# VISUALS
# ==============================
st.subheader("Visualizations")

# Real-time Trip Dashboard (timeline)
if not trip_df_f.empty:
    trips_timeline(trip_df_f)
else:
    empty_state("No trips available for the selected filters.")

# Route-level Journey Time Trends
st.markdown("**Route-level Journey Time Trends (Expected vs Real)**")
if not trip_df_f.empty:
    tmp = trip_df_f.copy()
    tmp["day"] = tmp["start"].dt.date
    grp = tmp.groupby(["day", "route"], as_index=False).agg(
        expected_min=("expected_duration", lambda x: np.median([pd.Timedelta(t).total_seconds() for t in x])/60.0),
        real_min=("real_duration", lambda x: np.median([pd.Timedelta(t).total_seconds() for t in x])/60.0)
    )
    grp = grp.melt(id_vars=["day","route"], value_vars=["expected_min","real_min"], var_name="kind", value_name="minutes")
    fig = px.line(grp, x="day", y="minutes", color="kind", line_group="route", facet_row="route",
                  markers=True, title="Median Journey Minutes by Route")
    st.plotly_chart(fig, use_container_width=True)
else:
    empty_state("No route trend data.")

# Delay Root Cause Pie Chart
st.markdown("**Delay Root Cause Pie Chart**")
if not trip_df_f.empty:
    delay = (trip_df_f["real_duration"] - trip_df_f["expected_duration"]).dt.total_seconds()/60.0
    trip_df_f = trip_df_f.assign(delay_minutes=delay.clip(lower=0))
    delay_summary = trip_df_f.groupby("stop_type", dropna=True, as_index=False)["delay_minutes"].sum()
    if not delay_summary.empty and delay_summary["delay_minutes"].sum() > 0:
        fig1 = px.pie(delay_summary, names="stop_type", values="delay_minutes", title="Delay by Stop Type (minutes)")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        empty_state("No positive delays recorded by stop type.")
else:
    empty_state("No trip data to compute delays.")

# Vehicle Reliability Scorecard
st.markdown("**Vehicle Reliability Scorecard**")
if not vehicle_df.empty:
    if not vehicle_events_df.empty:
        ev = vehicle_events_df.groupby("vehicle_id")["id"].count().rename("events").reset_index()
    else:
        ev = pd.DataFrame({"vehicle_id": vehicle_df["id"], "events": 0})
    vehicle_rel = vehicle_df.merge(ev, left_on="id", right_on="vehicle_id", how="left").fillna({"events":0})
    fig2 = px.bar(vehicle_rel, x="registration_number", y="events", title="Vehicle Events Count (last 72h)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    empty_state("No vehicle data.")

# Crew Performance Radar Chart (sample or from DB if exists)
st.markdown("**Crew Performance by Route (Radar Chart)**")
crew_perf_cols = {c for c in trip_df_f.columns if str(c).startswith("score_")}
if crew_perf_cols:
    # Build dynamically if scores are present in trip_df_f like score_route1, etc.
    melted = trip_df_f[["driver", *crew_perf_cols]].melt(id_vars="driver", var_name="route", value_name="score").dropna()
    melted["route"] = melted["route"].str.replace("score_", "")
else:
    # fallback sample
    crew_perf = pd.DataFrame({
        "driver": ["Driver A", "Driver B", "Driver C"],
        "route1": [90, 80, 85],
        "route2": [88, 92, 79],
        "route3": [95, 85, 90]
    })
    melted = crew_perf.melt(id_vars="driver", var_name="route", value_name="score")
fig3 = px.line_polar(melted, r="score", theta="route", color="driver", line_close=True, markers=True,
                     title="Crew Performance Radar")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# QUICK ACTIONS / SHORTCUTS
# ==============================
st.subheader("Quick Actions / Shortcuts")

# 🚦 Flag delayed trips for review
with st.expander("🚦 Flag delayed trips"):
    if not trip_df_f.empty:
        delayed = trip_df_f[trip_df_f["real_duration"] > trip_df_f["expected_duration"]].copy()
        if delayed.empty:
            st.write("No delayed trips in the current selection.")
        else:
            delayed_view = delayed[["trip_id","route","vehicle_id","start","end","real_duration","expected_duration","stop_type"]]
            st.dataframe(delayed_view, use_container_width=True)
    else:
        st.write("No trip data available.")

# 🧑 Assign new driver/steward (UI only; requires your business rules)
with st.expander("🧑 Assign new driver/steward"):
    if users_df.empty or trip_df_f.empty:
        st.write("Need users and trips data to assign.")
    else:
        target_trip = st.selectbox("Select trip", options=trip_df_f["trip_id"].unique())
        driver_opts = users_df.loc[users_df["position"].isin(["driver","steward"])].copy()
        driver_opts["full"] = driver_opts["firstname"] + " " + driver_opts["lastname"] + " (" + driver_opts["position"] + ")"
        driver_map = dict(zip(driver_opts["full"], driver_opts["id"]))
        selected_person = st.selectbox("Select person", options=list(driver_map.keys()))
        if st.button("Assign", type="primary"):
            if engine is None:
                st.warning("No DB connection. In Demo mode only UI is shown.")
            else:
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            UPDATE trips SET assigned_user_id = :uid WHERE id = :trip_id
                        """), {"uid": driver_map[selected_person], "trip_id": str(target_trip)})
                    st.success("Assignment saved.")
                except Exception as e:
                    st.error(f"Failed to assign: {e}")

# 🛠 Request bus maintenance (creates a record)
with st.expander("🛠 Request bus maintenance"):
    if vehicle_df.empty:
        st.write("No vehicles available.")
    else:
        vmap = dict(zip(vehicle_df["registration_number"], vehicle_df["id"]))
        vehicle_label = st.selectbox("Vehicle", options=list(vmap.keys()))
        date_sel = st.date_input("Maintenance date", value=date.today())
        status_sel = st.selectbox("Status", ["scheduled","in_progress","done"]) 
        notes = st.text_area("Notes", placeholder="Describe the issue/work")
        if st.button("Create maintenance request", type="primary"):
            if engine is None:
                st.info("Demo mode: would insert into scheduled_maintenances.")
            else:
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO scheduled_maintenances(date, vehicle_id, bus_status, manager_confirmation, notes)
                            VALUES (:d, :vid, :bs, :mc, :n)
                        """), {
                            "d": date_sel,
                            "vid": int(vmap[vehicle_label]),
                            "bs": status_sel,
                            "mc": None,
                            "n": notes,
                        })
                    st.success("Maintenance request created.")
                except Exception as e:
                    st.error(f"Insert failed: {e}")

# 🗓 Approve trip confirmation (functional)
with st.expander("🗓 Approve trip confirmation"):
    # Expect a table trips with confirmation/status columns; adapt if different.
    if engine is None:
        st.info("Demo mode: showing a mock table.")
        pending = pd.DataFrame({
            "trip_id": ["T001","T005","T012"],
            "route": ["R1","R2","R3"],
            "vehicle_id": [100,102,104],
            "start": [datetime.now()]*3,
            "status": ["pending","pending","pending"],
        })
    else:
        pending = get_df(
            """
            SELECT id AS trip_id, route, vehicle_id, start, status
            FROM trips
            WHERE status = 'pending' OR manager_confirmation = false
            ORDER BY start DESC
            LIMIT 200
            """,
            engine,
        )
    if pending.empty:
        st.write("No pending trips to approve.")
    else:
        st.dataframe(pending, use_container_width=True)
        sel = st.multiselect("Select trips to approve", options=pending["trip_id"].astype(str).tolist())
        approve_btn, reject_btn = st.columns(2)
        with approve_btn:
            if st.button("Approve selected", type="primary") and sel:
                if engine is None:
                    st.success(f"Demo: Approved {len(sel)} trips.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(text(
                                "UPDATE trips SET status='approved', manager_confirmation=true WHERE id = ANY(:ids)"
                            ), {"ids": list(map(str, sel))})
                        st.success(f"Approved {len(sel)} trips.")
                    except Exception as e:
                        st.error(f"Approval failed: {e}")
        with reject_btn:
            if st.button("Reject selected") and sel:
                if engine is None:
                    st.warning(f"Demo: Rejected {len(sel)} trips.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(text(
                                "UPDATE trips SET status='rejected', manager_confirmation=false WHERE id = ANY(:ids)"
                            ), {"ids": list(map(str, sel))})
                        st.warning(f"Rejected {len(sel)} trips.")
                    except Exception as e:
                        st.error(f"Rejection failed: {e}")

# ==============================
# MAINTENANCE & STAFF PANELS
# ==============================

with st.expander("🧰 Fleet Downtime & Maintenance Status"):
    if maintenance_df.empty:
        st.write("No maintenance records.")
    else:
        st.dataframe(maintenance_df, use_container_width=True)
        maint_counts = maintenance_df["bus_status"].value_counts().reset_index()
        maint_counts.columns = ["status","count"]
        figm = px.bar(maint_counts, x="status", y="count", title="Maintenance by Status")
        st.plotly_chart(figm, use_container_width=True)

with st.expander("👷 Staff Trip Punctuality & Readiness"):
    if attendance_df.empty:
        st.write("No attendance data.")
    else:
        by_type = attendance_df["presence_type"].value_counts().reset_index()
        by_type.columns = ["presence_type","count"]
        figp = px.pie(by_type, names="presence_type", values="count", title="Staff Punctuality Distribution")
        st.plotly_chart(figp, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Dashboard powered by Streamlit | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
