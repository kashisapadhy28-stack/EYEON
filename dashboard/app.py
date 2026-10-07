"""
EYEON - Intelligent Security Monitoring Dashboard
Streamlit-based operator assistance interface.
"""

import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.alerts.alert_manager import AlertManager, AlertState
from src.database.db import DatabaseManager
from src.risk.risk_engine import RiskEngine, RiskLevel
from src.risk.risk_video import process_video
from src.utils.config_loader import load_config, get_path


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EYEON — Intelligent Security Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark glassmorphism aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3440;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .status-badge-high {
        background-color: #e74c3c;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-med {
        background-color: #f39c12;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-low {
        background-color: #2ecc71;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    return DatabaseManager()


@st.cache_resource
def get_alert_mgr():
    return AlertManager(db_manager=get_db())


def main():
    db = get_db()
    alert_mgr = get_alert_mgr()
    cfg = load_config()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    st.sidebar.image("https://img.icons8.com/color/96/visible--v1.png", width=64)
    st.sidebar.title("EYEON Operations")
    st.sidebar.caption("Intelligent Object & Person Tracking System")

    st.sidebar.divider()

    counts = db.get_event_counts()
    st.sidebar.markdown("### 📊 Live System Status")
    col_sb1, col_sb2 = st.sidebar.columns(2)
    col_sb1.metric("High Risk", counts.get("HIGH", 0))
    col_sb2.metric("Medium Risk", counts.get("MEDIUM", 0))

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Quick Settings")
    conf_person = st.sidebar.slider("Person Detection Threshold", 0.1, 1.0, float(cfg["detection"]["confidence"]["person"]))
    conf_obj = st.sidebar.slider("Object Detection Threshold", 0.1, 1.0, float(cfg["detection"]["confidence"]["object"]))

    # -----------------------------------------------------------------------
    # Header Banner
    # -----------------------------------------------------------------------
    st.title("👁️ EYEON Security Operations Dashboard")
    st.markdown("*Real-time AI person-object tracking, zone analysis, and explainable risk scoring.*")

    # -----------------------------------------------------------------------
    # Main Navigation Tabs
    # -----------------------------------------------------------------------
    tab_pipeline, tab_events, tab_snapshots, tab_analytics, tab_config = st.tabs([
        "📹 Video Pipeline Runner",
        "🚨 Security Event Log",
        "🖼️ Alert Snapshots",
        "📈 Analytics & Risk Metrics",
        "⚙️ System Configuration",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Video Pipeline Runner
    # -----------------------------------------------------------------------
    with tab_pipeline:
        st.subheader("Process Monitored Video Feed")
        videos_dir = Path("data/videos")
        video_files = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
        video_options = [v.name for v in video_files]

        if not video_options:
            st.warning("No test videos found in `data/videos/`. Please upload or place `.mp4` files in `data/videos/`.")
        else:
            selected_video_name = st.selectbox("Select Target Video Stream", video_options)
            selected_video_path = videos_dir / selected_video_name

            col_p1, col_p2 = st.columns([2, 1])

            with col_p1:
                st.video(str(selected_video_path))

            with col_p2:
                st.markdown("#### Pipeline Execution")
                max_f = st.number_input("Max Frames to Process (0 = full video)", min_value=0, max_value=5000, value=150, step=50)
                max_f_arg = max_f if max_f > 0 else None

                if st.button("▶️ Run Full AI Pipeline", type="primary"):
                    out_path = Path("data/processed") / f"dashboard_out_{selected_video_name}"
                    with st.spinner("Processing video stream with Detection, Tracking, Association, Zones & Risk Engine..."):
                        start_t = time.time()
                        process_video(
                            input_path=selected_video_path,
                            output_path=out_path,
                            display=False,
                            max_frames=max_f_arg,
                        )
                        elapsed = time.time() - start_t

                    st.success(f"Processing complete in {elapsed:.2f}s!")
                    if out_path.exists():
                        st.markdown("#### Annotated Output Stream")
                        st.video(str(out_path))

    # -----------------------------------------------------------------------
    # Tab 2: Security Event Log
    # -----------------------------------------------------------------------
    with tab_events:
        st.subheader("Security Alerts & Incident Register")

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filter_level = st.selectbox("Filter Risk Level", ["ALL", "HIGH", "MEDIUM", "LOW"])
        with col_f2:
            st.write("")

        level_arg = None if filter_level == "ALL" else filter_level
        events = db.get_recent_events(limit=100, level=level_arg)

        if not events:
            st.info("No security events logged matching current criteria.")
        else:
            # Render interactive table
            df_events = pd.DataFrame(events)
            display_cols = ["alert_id", "datetime_str", "person_track_id", "risk_score", "risk_level", "status", "reasons", "zones"]
            df_display = df_events[display_cols].copy()
            df_display.columns = ["Alert ID", "Timestamp", "Person ID", "Score", "Risk Level", "Status", "Reasons", "Zones"]

            st.dataframe(df_display, use_container_state=True)

            # Interactive Alert State Management
            st.markdown("#### 🛠️ Manage Alert Status")
            col_m1, col_m2, col_m3 = st.columns([2, 1, 1])

            with col_m1:
                selected_alert_id = st.selectbox("Select Alert ID to Update", df_display["Alert ID"].tolist())

            with col_m2:
                if st.button("✅ Mark Acknowledged"):
                    if db.update_event_status(selected_alert_id, AlertState.ACKNOWLEDGED.value):
                        st.success(f"Alert {selected_alert_id} marked as ACKNOWLEDGED")
                        st.rerun()

            with col_m3:
                if st.button("🔒 Mark Resolved"):
                    if db.update_event_status(selected_alert_id, AlertState.RESOLVED.value):
                        st.success(f"Alert {selected_alert_id} marked as RESOLVED")
                        st.rerun()

    # -----------------------------------------------------------------------
    # Tab 3: Alert Snapshots
    # -----------------------------------------------------------------------
    with tab_snapshots:
        st.subheader("High-Risk Event Snapshot Gallery")
        snapshot_dir = Path("data/alerts")

        if not snapshot_dir.exists() or not list(snapshot_dir.glob("*.jpg")):
            st.info("No alert snapshots captured yet. Run the pipeline on a video to trigger alerts.")
        else:
            snapshots = list(snapshot_dir.glob("*.jpg"))
            cols = st.columns(3)
            for idx, img_path in enumerate(snapshots[:12]):
                with cols[idx % 3]:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Snapshot: {img_path.name}", use_column_width=True)

    # -----------------------------------------------------------------------
    # Tab 4: Analytics & Metrics
    # -----------------------------------------------------------------------
    with tab_analytics:
        st.subheader("System Risk & Zone Analytics")

        c1, c2, c3, c4 = st.columns(4)
        total_ev = sum(counts.values())
        c1.metric("Total Events", total_ev)
        c2.metric("HIGH Alerts", counts.get("HIGH", 0))
        c3.metric("MEDIUM Alerts", counts.get("MEDIUM", 0))
        c4.metric("LOW Events", counts.get("LOW", 0))

        st.divider()

        if total_ev > 0:
            st.markdown("#### Risk Level Distribution")
            chart_data = pd.DataFrame({
                "Risk Level": list(counts.keys()),
                "Count": list(counts.values()),
            }).set_index("Risk Level")
            st.bar_chart(chart_data)
        else:
            st.info("Run detection pipeline to generate analytics data.")

    # -----------------------------------------------------------------------
    # Tab 5: Configuration
    # -----------------------------------------------------------------------
    with tab_config:
        st.subheader("System Configuration (`config.yaml`)")
        st.json(cfg)


if __name__ == "__main__":
    main()
