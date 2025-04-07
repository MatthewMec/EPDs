import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from pulp import LpProblem, LpVariable, LpMaximize, lpSum

# Set page config
st.set_page_config(
    page_title="Bull Breeding Analysis Dashboard",
    page_icon="🐄",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>Bull Breeding Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("This dashboard provides analysis and optimization for bull breeding based on EPD (Expected Progeny Difference) data.")

# Sidebar for navigation and filters
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", [
    "Data Overview", 
    "Breeding Criteria Analysis", 
    "Cow Performance", 
    "Bull Analysis", 
    "Optimization Results"
])

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the data
        epds = pl.read_csv('2025Bull.csv', ignore_errors=True).with_row_index().drop(['Sex', 'DOB', 'BrdCds'])
        
        # Process data similar to the original code
        epds = epds.with_columns(mean=pl.sum_horizontal(['HB', 'GM']) / 2)
        
        epds = (epds.with_columns([
            pl.col("AnimalID").str.split_exact(" - ", 1).
            alias("split_column")]).
            with_columns(pl.col('split_column')).
            unnest('split_column').
            rename({'field_0': 'cow', 'field_1': 'bull'})
        )
        
        # Add target and comparison columns
        _epds_with_target = epds.with_columns(((pl.col('HB') + pl.col('GM')) / 2).alias('target'))
        
        mean_target = _epds_with_target.select(pl.mean('target')).item()
        mean_hb = _epds_with_target.select(pl.mean('HB')).item()
        mean_gm = _epds_with_target.select(pl.mean('GM')).item()
        
        _epds_with_target = _epds_with_target.with_columns(
            pl.when(pl.col('target') > mean_target)
            .then(pl.lit("above"))
            .otherwise(pl.lit("below"))
            .alias('comparison')
        )
        
        _epds_with_target = _epds_with_target.with_columns(
            pl.when(pl.col('HB') > mean_hb)
            .then(pl.lit("above"))
            .otherwise(pl.lit("below"))
            .alias('comparison_hb')
        )
        
        _epds_with_target = _epds_with_target.with_columns(
            pl.when(pl.col('GM') > mean_gm)
            .then(pl.lit("above"))
            .otherwise(pl.lit("below"))
            .alias('comparison_gm')
        )
        
        return _epds_with_target
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data if actual data can't be loaded
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data if the real data file is not available"""
    st.warning("Using sample data. Please upload the '2025Bull.csv' file for actual analysis.")
    
    # Create sample data with similar structure
    data = {
        "AnimalID": [f"Cow{i} - Bull{j}" for i in range(1, 10) for j in range(1, 5)],
        "HB": np.random.normal(0.5, 0.2, 36),
        "GM": np.random.normal(0.6, 0.2, 36),
        "ME": np.random.normal(-1.5, 0.5, 36),
        "CED": np.random.normal(10, 3, 36),
        "Milk": np.random.normal(25, 5, 36),
        "CEM": np.random.normal(6, 2, 36),
        "Marb": np.random.normal(0.4, 0.1, 36),
        "STAY": np.random.normal(16, 3, 36),
        "REA": np.random.normal(0.4, 0.2, 36),
        "CW": np.random.normal(30, 10, 36),
        "YG": np.random.normal(0.1, 0.05, 36)
    }
    
    _df = pl.DataFrame(data).with_row_index()
    
    # Process like the actual data
    _df = _df.with_columns(mean=pl.sum_horizontal(['HB', 'GM']) / 2)
    
    _df = (_df.with_columns([
        pl.col("AnimalID").str.split_exact(" - ", 1).
        alias("split_column")]).
        with_columns(pl.col('split_column')).
        unnest('split_column').
        rename({'field_0': 'cow', 'field_1': 'bull'})
    )
    
    # Add target and comparison columns
    _df_with_target = _df.with_columns(((pl.col('HB') + pl.col('GM')) / 2).alias('target'))
    
    mean_target = _df_with_target.select(pl.mean('target')).item()
    mean_hb = _df_with_target.select(pl.mean('HB')).item()
    mean_gm = _df_with_target.select(pl.mean('GM')).item()
    
    _df_with_target = _df_with_target.with_columns(
        pl.when(pl.col('target') > mean_target)
        .then(pl.lit("above"))
        .otherwise(pl.lit("below"))
        .alias('comparison')
    )
    
    _df_with_target = _df_with_target.with_columns(
        pl.when(pl.col('HB') > mean_hb)
        .then(pl.lit("above"))
        .otherwise(pl.lit("below"))
        .alias('comparison_hb')
    )
    
    _df_with_target = _df_with_target.with_columns(
        pl.when(pl.col('GM') > mean_gm)
        .then(pl.lit("above"))
        .otherwise(pl.lit("below"))
        .alias('comparison_gm')
    )
    
    return _df_with_target

# Allow uploading a file
uploaded_file = st.sidebar.file_uploader("Upload your bull data (CSV)", type="csv")
if uploaded_file is not None:
    # Save the uploaded file
    with open("2025Bull.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully!")

# Load the data
_df = load_data()

# Define the criteria similar to your original code
default_criteria = {
    "ME": -1.0,  # Convert to float
    "CED": 12,
    "Milk": 26,
    "CEM": 7,
    "Marb": 0.4,
    "STAY": 17
}

# Sidebar for adjusting criteria
st.sidebar.markdown("### Breeding Criteria")
criteria = {}
with st.sidebar.expander("Adjust Breeding Criteria", expanded=False):
    # Fix: Ensure all values are of the same type (float)
    criteria["ME"] = st.slider("ME (≤)", float(-2.0), float(2.0), float(default_criteria["ME"]), float(0.1))
    criteria["CED"] = st.slider("CED (≥)", 0, 20, default_criteria["CED"])
    criteria["Milk"] = st.slider("Milk (≥)", 0, 50, default_criteria["Milk"])
    criteria["CEM"] = st.slider("CEM (≥)", 0, 15, default_criteria["CEM"])
    criteria["Marb"] = st.slider("Marb (≥)", float(0.0), float(1.0), float(default_criteria["Marb"]), float(0.1))
    criteria["STAY"] = st.slider("STAY (≥)", 0, 30, default_criteria["STAY"])

# Apply criteria
@st.cache_data
def apply_criteria(_df, criteria):
    return _df.filter(
        (pl.col("ME") <= criteria["ME"]) &
        (pl.col("CED") >= criteria["CED"]) &
        (pl.col("Milk") >= criteria["Milk"]) &
        (pl.col("CEM") >= criteria["CEM"]) &
        (pl.col("Marb") >= criteria["Marb"]) &
        (pl.col("STAY") >= criteria["STAY"])
    )

_df_filtered = apply_criteria(_df, criteria)

# Calculate summary statistics
def calculate_stats(_df):
    stats = {
        "total_cows": _df.select(pl.col('cow').n_unique()).item(),
        "total_bulls": _df.select(pl.col('bull').n_unique()).item(),
        "mean_hb": _df.select(pl.mean('HB')).item(),
        "mean_gm": _df.select(pl.mean('GM')).item(),
        "mean_target": _df.select(pl.mean('target')).item(),
    }
    return stats

stats_original = calculate_stats(_df)
stats_filtered = calculate_stats(_df_filtered)

# Run the optimization
@st.cache_data
def run_optimization(_epds_with_target):
    try:
        # Define constraints
        bull_limit = len(_epds_with_target['cow'].unique()) // 5

        # Initialize the problem
        prob = LpProblem("Breeding_Optimization", LpMaximize)

        # Create decision variables
        n = len(_epds_with_target)
        bulls = _epds_with_target['bull'].unique().to_list()
        vars = {
            (i, j): LpVariable(f"x_{i}_{j}", cat=LPBinary)
            for i in range(n)
            for j in range(len(bulls))
        }

        # Objective function: Maximize total target based on 'above' or 'below'
        prob += lpSum(
            vars[i, j] * _epds_with_target['target'][i]
            for i in range(n)
            for j in range(len(bulls))
            if _epds_with_target['comparison'][i] == 'above'
        )

        # Add constraints: Each bull can breed with at most 1/5 of the cows
        for bull_idx, bull in enumerate(bulls):
            prob += lpSum(
                vars[i, bull_idx]
                for i in range(n)
                if _epds_with_target['bull'][i] == bull
            ) <= bull_limit

        # Constraints: Each cow breeds only once
        cows = _epds_with_target['cow'].unique().to_list()
        for cow_idx, cow in enumerate(cows):
            prob += lpSum(
                vars[i, j]
                for i in range(n)
                if _epds_with_target['cow'][i] == cow
                for j in range(len(bulls))
            ) <= 1

        # Solve the problem
        prob.solve()

        # Extract the results
        result = []
        for i in range(n):
            for j in range(len(bulls)):
                if vars[i, j].value() == 1:
                    result.append({
                        "cow": _epds_with_target['cow'][i], 
                        "bull": bulls[j]
                    })

        if not result:
            return None
            
        result__df = pl.DataFrame(result)
        
        # Count bulls
        bull_counts = result__df.group_by("bull").agg(
            pl.len().alias("count")
        ).sort("count", descending=True)
        
        return {
            "pairings": result__df,
            "bull_counts": bull_counts
        }
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return None

# Apply pages based on selection
if page == "Data Overview":
    st.markdown("<div class='sub-header'>Data Overview</div>", unsafe_allow_html=True)
    
    # Display summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cows", stats_original["total_cows"])
    with col2:
        st.metric("Total Bulls", stats_original["total_bulls"])
    with col3:
        st.metric("Total Pairings", len(_df))
    
    # Show data sample
    st.markdown("<div class='sub-header'>Data Sample</div>", unsafe_allow_html=True)
    st.dataframe(_df.head(10))
    
    # Distribution plots
    st.markdown("<div class='sub-header'>Key Metric Distributions</div>", unsafe_allow_html=True)
    metric_option = st.selectbox(
        "Select metric to visualize:", 
        ["HB", "GM", "target", "ME", "CED", "Milk", "CEM", "Marb", "STAY"]
    )
    
    fig = px.histogram(_df.to_pandas(), x=metric_option, 
                       title=f"Distribution of {metric_option}",
                       color_discrete_sequence=['#3498db'])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Breeding Criteria Analysis":
    st.markdown("<div class='sub-header'>Breeding Criteria Analysis</div>", unsafe_allow_html=True)
    
    # Show criteria impact
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cows Meeting Criteria", 
                 stats_filtered["total_cows"], 
                 delta=stats_filtered["total_cows"] - stats_original["total_cows"])
    with col2:
        if stats_original["total_cows"] > 0:
            percentage = (stats_filtered["total_cows"] / stats_original["total_cows"]) * 100
            st.metric("Percentage of Cows Meeting Criteria", f"{percentage:.1f}%")
    
    # List cows that don't meet criteria
    cows_all = set(_df['cow'].unique())
    cows_meeting = set(_df_filtered['cow'].unique())
    cows_failing = cows_all - cows_meeting
    
    if cows_failing:
        st.markdown("<div class='sub-header'>Cows Not Meeting Criteria</div>", unsafe_allow_html=True)
        
        # Group by specific failing criteria
        failing_reasons = []
        for cow in cows_failing:
            cow_data = _df.filter(pl.col("cow") == cow)
            
            fail_reasons = []
            if cow_data.filter(pl.col("ME") <= criteria["ME"]).is_empty():
                fail_reasons.append("ME")
            if cow_data.filter(pl.col("CED") >= criteria["CED"]).is_empty():
                fail_reasons.append("CED")
            if cow_data.filter(pl.col("Milk") >= criteria["Milk"]).is_empty():
                fail_reasons.append("Milk")
            if cow_data.filter(pl.col("CEM") >= criteria["CEM"]).is_empty():
                fail_reasons.append("CEM")
            if cow_data.filter(pl.col("Marb") >= criteria["Marb"]).is_empty():
                fail_reasons.append("Marb")
            if cow_data.filter(pl.col("STAY") >= criteria["STAY"]).is_empty():
                fail_reasons.append("STAY")
            
            failing_reasons.append({
                "cow": cow,
                "failing_criteria": ", ".join(fail_reasons)
            })
        
        fail__df = pl.DataFrame(failing_reasons)
        st.dataframe(fail__df, use_container_width=True)
        
        # Visualize failing criteria distribution
        if failing_reasons:
            all_fails = []
            for item in failing_reasons:
                criteria_list = item["failing_criteria"].split(", ")
                for c in criteria_list:
                    all_fails.append(c)
            
            fail_counts = {}
            for c in all_fails:
                if c in fail_counts:
                    fail_counts[c] += 1
                else:
                    fail_counts[c] = 1
            
            criteria__df = pd.DataFrame({"Criteria": list(fail_counts.keys()), 
                                        "Count": list(fail_counts.values())})
            
            fig = px.bar(criteria__df, x="Criteria", y="Count", 
                         title="Distribution of Failing Criteria",
                         color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)

elif page == "Cow Performance":
    st.markdown("<div class='sub-header'>Cow Performance Analysis</div>", unsafe_allow_html=True)
    
    # Calculate metrics by cow
    mean_by_cow = _df.group_by("cow").agg([
        pl.col("HB").mean().alias("mean_hb"),
        pl.col("GM").mean().alias("mean_gm"),
        pl.col("target").mean().alias("mean_target")
    ]).sort("mean_target", descending=True)
    
    # Visualization options
    viz_option = st.radio(
        "Select visualization:",
        ["Top Performing Cows", "Cow Metric Comparison", "Detailed Cow Analysis"]
    )
    
    if viz_option == "Top Performing Cows":
        top_n = st.slider("Select number of top cows to show:", 5, 20, 10)
        
        fig = px.bar(mean_by_cow.head(top_n).to_pandas(), 
                     x="cow", y="mean_target", 
                     title=f"Top {top_n} Cows by Target Performance",
                     color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Cow Metric Comparison":
        fig = px.scatter(mean_by_cow.to_pandas(), x="mean_hb", y="mean_gm", 
                         hover_data=["cow", "mean_target"],
                         title="Cow Performance: HB vs GM",
                         color="mean_target",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Detailed Cow Analysis":
        selected_cow = st.selectbox("Select a cow for detailed analysis:", 
                                   sorted(_df['cow'].unique().to_list()))
        
        # Filter data for the selected cow
        cow_data = _df.filter(pl.col("cow") == selected_cow)
        
        # Create a radar chart for the cow's metrics
        metrics = ["HB", "GM", "ME", "CED", "Milk", "CEM", "Marb", "STAY"]
        cow_means = cow_data.select([pl.col(m).mean() for m in metrics]).to_dict(as_series=False)
        
        # Normalize the values for better visualization
        _df_means = _df.select([pl.col(m).mean() for m in metrics]).to_dict(as_series=False)
        
        radar_data = []
        for i, m in enumerate(metrics):
            radar_data.append({
                "metric": m,
                "value": cow_means[m][0],
                "population_mean": _df_means[m][0]
            })
        
        radar__df = pd.DataFrame(radar_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar__df["value"].tolist(),
            theta=radar__df["metric"].tolist(),
            fill='toself',
            name=selected_cow
        ))
        fig.add_trace(go.Scatterpolar(
            r=radar__df["population_mean"].tolist(),
            theta=radar__df["metric"].tolist(),
            fill='toself',
            name='Population Mean'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            showlegend=True,
            title=f"Metric Comparison for Cow {selected_cow}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show bulls used with this cow
        st.markdown(f"##### Bulls Used with Cow {selected_cow}")
        bull_data = cow_data.select(["bull", "HB", "GM", "target"]).sort("target", descending=True)
        st.dataframe(bull_data, use_container_width=True)

elif page == "Bull Analysis":
    st.markdown("<div class='sub-header'>Bull Analysis</div>", unsafe_allow_html=True)
    
    # Count bulls
    bull_counts = _df.group_by("bull").agg(
        pl.len().alias("count")
    ).sort("count", descending=True)
    
    # Calculate metrics by bull
    mean_by_bull = _df.group_by("bull").agg([
        pl.col("HB").mean().alias("mean_hb"),
        pl.col("GM").mean().alias("mean_gm"),
        pl.col("target").mean().alias("mean_target"),
        pl.len().alias("pairing_count")
    ]).sort("mean_target", descending=True)
    
    # Visualization options
    viz_option = st.radio(
        "Select visualization:",
        ["Bull Usage Count", "Top Performing Bulls", "Bull Metric Comparison"]
    )
    
    if viz_option == "Bull Usage Count":
        top_n = st.slider("Select number of bulls to show:", 5, 20, 10)
        
        fig = px.bar(bull_counts.head(top_n).to_pandas(), 
                     x="bull", y="count", 
                     title=f"Top {top_n} Bulls by Usage Count",
                     color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Top Performing Bulls":
        top_n = st.slider("Select number of top bulls to show:", 5, 20, 10)
        min_pairings = st.slider("Minimum number of pairings:", 1, 10, 3)
        
        filtered_bulls = mean_by_bull.filter(pl.col("pairing_count") >= min_pairings)
        
        fig = px.bar(filtered_bulls.head(top_n).to_pandas(), 
                     x="bull", y="mean_target", 
                     title=f"Top {top_n} Bulls by Target Performance (min {min_pairings} pairings)",
                     color="pairing_count",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_option == "Bull Metric Comparison":
        min_pairings = st.slider("Minimum number of pairings for comparison:", 1, 10, 3)
        
        filtered_bulls = mean_by_bull.filter(pl.col("pairing_count") >= min_pairings)
        
        fig = px.scatter(filtered_bulls.to_pandas(), 
                         x="mean_hb", y="mean_gm", 
                         size="pairing_count",
                         hover_data=["bull", "mean_target", "pairing_count"],
                         title="Bull Performance: HB vs GM",
                         color="mean_target",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Optimization Results":
    st.markdown("<div class='sub-header'>Breeding Optimization Results</div>", unsafe_allow_html=True)
    
    # Run optimization
    with st.spinner("Running breeding optimization..."):
        optimization_results = run_optimization(_df_filtered)
    
    if optimization_results:
        # Show results
        st.markdown("##### Optimal Breeding Pairs")
        st.dataframe(optimization_results["pairings"], use_container_width=True)
        
        # Show bull distribution
        st.markdown("##### Bull Usage in Optimal Solution")
        st.dataframe(optimization_results["bull_counts"], use_container_width=True)
        
        # Visualize bull usage
        fig = px.bar(optimization_results["bull_counts"].to_pandas(), 
                     x="bull", y="count", 
                     title="Bull Usage Count in Optimal Solution",
                     color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        csv = optimization_results["pairings"].to_pandas().to_csv(index=False)
        st.download_button(
            label="Download Optimal Breeding Pairs",
            data=csv,
            file_name="optimal_breeding_pairs.csv",
            mime="text/csv",
        )
    else:
        st.warning("Could not find an optimal solution with the current criteria. Try adjusting your criteria.")

# Footer
st.markdown("---")
st.markdown("Bull Breeding Analysis Dashboard | Created with Streamlit")