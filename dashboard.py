import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
METRICS_DIR = Path("metrics/data")


@st.cache_data
def load_all_metrics():
    """Load all JSON files from metrics/data directory and aggregate into DataFrame"""
    all_requests = []
    
    if not METRICS_DIR.exists():
        st.error(f"Metrics directory not found: {METRICS_DIR}")
        return pd.DataFrame()
    
    json_files = list(METRICS_DIR.glob("*.json"))
    
    if not json_files:
        st.warning(f"No JSON files found in {METRICS_DIR}")
        return pd.DataFrame()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            run_hash = data.get("run_hash", "")
            session_start = data.get("session_start", "")
            model_path = data.get("model_path", "")
            batch_size = data.get("batch_size", None)
            batch_wait_time = data.get("batch_wait_time", None)
            
            # Process each request
            requests = data.get("requests", [])
            for req in requests:
                req_row = req.copy()
                req_row["run_hash"] = run_hash
                req_row["session_start"] = session_start
                req_row["model_path"] = model_path
                req_row["batch_size_config"] = batch_size
                req_row["batch_wait_time"] = batch_wait_time
                req_row["source_file"] = json_file.name
                all_requests.append(req_row)
        except Exception as e:
            st.warning(f"Error loading {json_file.name}: {str(e)}")
            continue
    
    if not all_requests:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_requests)
    
    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Convert session_start to datetime
    if "session_start" in df.columns:
        df["session_start"] = pd.to_datetime(df["session_start"])
    
    return df


def filter_data(df, search_text, date_range, model_paths, batch_sizes, eval_score, temp_range):
    """Filter dataframe based on search and filter criteria"""
    filtered_df = df.copy()
    
    # Text search
    if search_text:
        search_cols = ["job_id", "gene_id", "run_hash", "model_path"]
        mask = pd.Series([False] * len(filtered_df))
        for col in search_cols:
            if col in filtered_df.columns:
                mask |= filtered_df[col].astype(str).str.contains(search_text, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Date range filter
    if date_range and len(date_range) == 2 and "timestamp" in filtered_df.columns:
        start_date, end_date = date_range
        if start_date:
            filtered_df = filtered_df[filtered_df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df["timestamp"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    
    # Model path filter
    if model_paths:
        filtered_df = filtered_df[filtered_df["model_path"].isin(model_paths)]
    
    # Batch size filter
    if batch_sizes:
        filtered_df = filtered_df[filtered_df["batch_size_config"].isin(batch_sizes)]
    
    # Evaluation score filter
    if eval_score != "All":
        score_val = float(eval_score)
        filtered_df = filtered_df[filtered_df["evaluation_score"] == score_val]
    
    # Temperature range filter
    if temp_range and "temperature" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["temperature"] >= temp_range[0]) &
            (filtered_df["temperature"] <= temp_range[1])
        ]
    
    return filtered_df


def main():
    st.title("📊 LLM Inference Metrics Dashboard")
    st.markdown("Visualize and explore metrics from your LLM inference runs")
    
    # Load data
    with st.spinner("Loading metrics data..."):
        df = load_all_metrics()
    
    if df.empty:
        st.error("No metrics data found. Please ensure JSON files exist in metrics/data folder.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Search & Filters")
    
    # Text search
    search_text = st.sidebar.text_input(
        "Search (job_id, gene_id, run_hash, model_path)",
        value="",
        help="Search across job_id, gene_id, run_hash, and model_path fields"
    )
    
    # Date range filter
    if "timestamp" in df.columns:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter requests by timestamp"
        )
    else:
        date_range = None
    
    # Model path filter
    if "model_path" in df.columns:
        model_paths = df["model_path"].unique().tolist()
        selected_models = st.sidebar.multiselect(
            "Model Path",
            options=model_paths,
            default=[],
            help="Select model paths to filter"
        )
    else:
        selected_models = []
    
    # Batch size filter
    if "batch_size_config" in df.columns:
        batch_sizes = sorted(df["batch_size_config"].dropna().unique().tolist())
        selected_batch_sizes = st.sidebar.multiselect(
            "Batch Size",
            options=batch_sizes,
            default=[],
            help="Select batch sizes to filter"
        )
    else:
        selected_batch_sizes = []
    
    # Evaluation score filter
    if "evaluation_score" in df.columns:
        eval_options = ["All"] + sorted(df["evaluation_score"].dropna().unique().tolist())
        eval_score = st.sidebar.selectbox(
            "Evaluation Score",
            options=eval_options,
            index=0,
            help="Filter by evaluation score"
        )
    else:
        eval_score = "All"
    
    # Temperature range filter
    if "temperature" in df.columns:
        temp_min = float(df["temperature"].min())
        temp_max = float(df["temperature"].max())
        temp_range = st.sidebar.slider(
            "Temperature Range",
            min_value=temp_min,
            max_value=temp_max,
            value=(temp_min, temp_max),
            help="Filter by temperature range"
        )
    else:
        temp_range = None
    
    # Apply filters
    filtered_df = filter_data(
        df,
        search_text,
        date_range if isinstance(date_range, tuple) else None,
        selected_models if selected_models else None,
        selected_batch_sizes if selected_batch_sizes else None,
        eval_score,
        temp_range
    )
    
    # Display filtered count
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Requests", len(filtered_df))
    if len(filtered_df) > 0 and "evaluation_score" in filtered_df.columns:
        success_rate = (filtered_df["evaluation_score"] == 1.0).sum() / len(filtered_df) * 100
        st.sidebar.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📋 Detailed View", "🔬 Comparisons"])
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust your search criteria.")
        return
    
    with tab1:
        st.header("Summary Statistics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if "e2e_latency_sec" in filtered_df.columns:
            with col1:
                st.metric("Mean Latency", f"{filtered_df['e2e_latency_sec'].mean():.2f}s")
            with col2:
                st.metric("Min Latency", f"{filtered_df['e2e_latency_sec'].min():.2f}s")
            with col3:
                st.metric("Max Latency", f"{filtered_df['e2e_latency_sec'].max():.2f}s")
        
        with col4:
            st.metric("Total Requests", len(filtered_df))
        
        if "evaluation_score" in filtered_df.columns:
            col5, col6, col7 = st.columns(3)
            success_count = (filtered_df["evaluation_score"] == 1.0).sum()
            with col5:
                st.metric("Success Count", success_count)
            with col6:
                st.metric("Failure Count", len(filtered_df) - success_count)
            with col7:
                st.metric("Success Rate", f"{(success_count / len(filtered_df) * 100):.1f}%")
        
        st.markdown("---")
        
        # Latency over time
        if "timestamp" in filtered_df.columns and "e2e_latency_sec" in filtered_df.columns:
            st.subheader("Latency Over Time")
            df_time = filtered_df[["timestamp", "e2e_latency_sec"]].sort_values("timestamp")
            
            fig = px.line(
                df_time,
                x="timestamp",
                y="e2e_latency_sec",
                title="End-to-End Latency Over Time",
                labels={"timestamp": "Time", "e2e_latency_sec": "Latency (seconds)"}
            )
            fig.update_traces(mode='lines+markers', marker_size=4)
            st.plotly_chart(fig, use_container_width=True)
        
        # Two column layout for charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Latency distribution
            if "e2e_latency_sec" in filtered_df.columns:
                st.subheader("Latency Distribution")
                fig_hist = px.histogram(
                    filtered_df,
                    x="e2e_latency_sec",
                    nbins=30,
                    title="Distribution of E2E Latency",
                    labels={"e2e_latency_sec": "Latency (seconds)", "count": "Frequency"}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Evaluation score distribution
            if "evaluation_score" in filtered_df.columns:
                st.subheader("Evaluation Score Distribution")
                score_counts = filtered_df["evaluation_score"].value_counts().sort_index()
                fig_score = px.bar(
                    x=score_counts.index.astype(str),
                    y=score_counts.values,
                    title="Evaluation Score Counts",
                    labels={"x": "Score", "y": "Count"}
                )
                st.plotly_chart(fig_score, use_container_width=True)
        
        with col_right:
            # Scatter plot: Prompt length vs latency
            if "prompt_length" in filtered_df.columns and "e2e_latency_sec" in filtered_df.columns:
                st.subheader("Prompt Length vs Latency")
                fig_scatter = px.scatter(
                    filtered_df,
                    x="prompt_length",
                    y="e2e_latency_sec",
                    color="evaluation_score" if "evaluation_score" in filtered_df.columns else None,
                    title="Prompt Length vs E2E Latency",
                    labels={"prompt_length": "Prompt Length", "e2e_latency_sec": "Latency (seconds)"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Scatter plot: Batch processing time vs latency
            if "batch_processing_time_sec" in filtered_df.columns and "e2e_latency_sec" in filtered_df.columns:
                st.subheader("Batch Processing Time vs E2E Latency")
                fig_scatter2 = px.scatter(
                    filtered_df,
                    x="batch_processing_time_sec",
                    y="e2e_latency_sec",
                    color="evaluation_score" if "evaluation_score" in filtered_df.columns else None,
                    title="Batch Processing Time vs E2E Latency",
                    labels={"batch_processing_time_sec": "Batch Processing Time (seconds)", 
                           "e2e_latency_sec": "E2E Latency (seconds)"}
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
    
    with tab2:
        st.header("Detailed Request Data")
        st.markdown(f"Showing {len(filtered_df)} requests")
        
        # Display dataframe with all columns
        display_cols = [
            "timestamp", "job_id", "gene_id", "run_hash", "model_path",
            "prompt_length", "max_new_tokens", "temperature", "top_p",
            "e2e_latency_sec", "batch_processing_time_sec", "batch_size",
            "queue_wait_time_sec", "evaluation_score"
        ]
        
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = filtered_df[available_cols].to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"metrics_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.header("Cross-Run Comparisons")
        
        if "run_hash" in filtered_df.columns:
            run_hashes = filtered_df["run_hash"].unique().tolist()
            selected_runs = st.multiselect(
                "Select Runs to Compare",
                options=run_hashes,
                default=run_hashes[:min(5, len(run_hashes))] if run_hashes else []
            )
            
            if selected_runs:
                compare_df = filtered_df[filtered_df["run_hash"].isin(selected_runs)]
                
                if "e2e_latency_sec" in compare_df.columns:
                    st.subheader("Latency Comparison by Run")
                    fig_compare = px.box(
                        compare_df,
                        x="run_hash",
                        y="e2e_latency_sec",
                        title="Latency Distribution by Run",
                        labels={"run_hash": "Run Hash", "e2e_latency_sec": "Latency (seconds)"}
                    )
                    fig_compare.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_compare, use_container_width=True)
                
                # Summary table by run
                st.subheader("Summary Statistics by Run")
                summary_cols = []
                if "e2e_latency_sec" in compare_df.columns:
                    summary_cols.append("e2e_latency_sec")
                if "evaluation_score" in compare_df.columns:
                    summary_cols.append("evaluation_score")
                
                if summary_cols:
                    summary = compare_df.groupby("run_hash")[summary_cols].agg({
                        "e2e_latency_sec": ["mean", "min", "max", "count"],
                        "evaluation_score": ["mean", "sum"]
                    } if "e2e_latency_sec" in summary_cols and "evaluation_score" in summary_cols else {})
                    st.dataframe(summary, use_container_width=True)
            else:
                st.info("Select at least one run to compare.")
        else:
            st.info("Run hash information not available for comparison.")


if __name__ == "__main__":
    main()

