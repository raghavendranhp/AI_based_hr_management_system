import streamlit as st
import pandas as pd
import os
from streamlit_option_menu import option_menu
from ui_components import render_global_dashboard, render_employee_search, render_ai_strategy
from recommendation_engine import HRRecommendationEngine

#configure the page layout to utilize the full screen width
st.set_page_config(
    page_title="Seshat AI HR System", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

#define data paths using relative routing
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(base_dir, 'data', 'processed')

@st.cache_data
@st.cache_data
def load_data():
    """
    loads all processed datasets into memory.
    halts the app gracefully if files are missing.
    """
    # define our directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # define explicit file paths
    hr_data_path = os.path.join(processed_dir, 'final_hr_dataset_processed.csv')
    insights_path = os.path.join(reports_dir, 'comprehensive_analytics_report.csv')
    dept_path = os.path.join(reports_dir, 'department_insight_summary.csv')
    
    # check if files exist before trying to load them
    missing_files = []
    if not os.path.exists(hr_data_path): missing_files.append(hr_data_path)
    if not os.path.exists(insights_path): missing_files.append(insights_path)
    if not os.path.exists(dept_path): missing_files.append(dept_path)
        
    if missing_files:
        st.error("🚨 **Critical Error: Missing Data Files**")
        st.write("The dashboard cannot start because it cannot find the following files:")
        for f in missing_files:
            st.code(f)
        st.info("Please ensure you have run the Day 2 and Day 3 Jupyter Notebooks, and that the files saved to these exact locations.")
        st.stop() # This prevents the app from continuing and throwing KeyErrors
        
    # if everything is found, load them
    hr_df = pd.read_csv(hr_data_path)
    insights_df = pd.read_csv(insights_path)
    dept_df = pd.read_csv(dept_path)
    
    return hr_df, insights_df, dept_df

# load datasets securely
hr_df, insights_df, dept_df = load_data()
@st.cache_resource
def load_ai_engine():
    """
    initializes the groq api client securely. cached to prevent re-initialization.
    """
    return HRRecommendationEngine()

# initialize the engine
engine = load_ai_engine()

#build the horizontal navigation menu using bootstrap icon strings
selected_tab = option_menu(
    menu_title=None, 
    options=["Global Dashboard", "Employee Search", "AI Strategy Engine"],
    icons=["bar-chart", "search", "lightbulb"], 
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#003366"},
    }
)


#routing logic based on horizontal menu selection
if selected_tab == "Global Dashboard":
    #call the modular layout function from ui_components.py
    render_global_dashboard(insights_df)
    
elif selected_tab == "Employee Search":
    #call the modular individual search layout function from ui_components.py
    render_employee_search(insights_df)
    
elif selected_tab == "AI Strategy Engine":
    # execute the ai generative interface
    render_ai_strategy(insights_df, dept_df, engine)