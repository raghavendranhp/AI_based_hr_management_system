import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set default seaborn styling for dashboard aesthetics
sns.set_theme(style="whitegrid")

def render_kpi_cards(insights_df):
    """
    renders top-level organizational key performance indicators in a row.
    """
    #calculate core organizational metrics
    total_employees = len(insights_df)
    critical_risks = insights_df['Critical_Flight_Risk'].sum()
    avg_kpi = insights_df['Forecasted_KPI_Score'].mean()
    high_flight_risk = len(insights_df[insights_df['Flight_Risk_Level'] == 'High'])

    #display metrics in a 4-column layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Workforce", f"{total_employees}")
    with col2:
        st.metric("Critical Flight Risks", f"{critical_risks}")
    with col3:
        st.metric("High Attrition Risk", f"{high_flight_risk}")
    with col4:
        st.metric("Avg Forecasted KPI", f"{avg_kpi:.1f}")

def render_department_health(insights_df):
    """
    renders a horizontal bar chart displaying high attrition risk by department.
    """
    st.subheader("Departmental Attrition Risk Profile")

    #calculate risk percentage per department
    dept_risk = insights_df.groupby('Department').apply(
        lambda x: (x['Flight_Risk_Level'] == 'High').sum() / len(x) * 100
    ).reset_index(name='High_Risk_Pct')
    
    #sort for visual hierarchy
    dept_risk = dept_risk.sort_values(by='High_Risk_Pct', ascending=False)

    #create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dept_risk, x='High_Risk_Pct', y='Department', palette='Reds_r', ax=ax)
    ax.set_xlabel('High Risk Employees (%)')
    ax.set_ylabel('')
    
    #render plot natively in streamlit
    st.pyplot(fig)

def render_9_box_grid(insights_df):
    """
    renders the human resources 9-box action grid showing performance vs flight risk.
    """
    st.subheader("9-Box Action Grid")

    #generate frequency matrix
    nine_box_matrix = pd.crosstab(
        insights_df['Flight_Risk_Level'], 
        insights_df['Performance_Level']
    )
    
    #ensure strict ordering for high risk at top and high performance at right
    nine_box_matrix = nine_box_matrix.reindex(
        index=['High', 'Medium', 'Low'], 
        columns=['Low', 'Medium', 'High'],
        fill_value=0
    )

    #create heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        nine_box_matrix, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd', 
        cbar=False, 
        linewidths=1, 
        linecolor='black', 
        ax=ax
    )
    ax.set_xlabel('Forecasted Performance Level', fontweight='bold')
    ax.set_ylabel('Flight Risk Level', fontweight='bold')
    
    st.pyplot(fig)

def render_global_dashboard(insights_df):
    """
    master function to orchestrate the global dashboard layout.
    """
    st.markdown("### Organizational Health Overview")
    st.markdown("---")
    
    #inject top kpi row
    render_kpi_cards(insights_df)
    
    st.markdown("---")
    
    #create a robust two-column layout for the charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        render_department_health(insights_df)
        
    with chart_col2:
        render_9_box_grid(insights_df)

def render_employee_profile_card(employee_row):
    """
    renders an executive summary metric card for a single selected employee.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    #calculate individual status parameters
    risk_pct = employee_row['Attrition_Probability'] * 100
    risk_level = employee_row['Flight_Risk_Level']
    perf_level = employee_row['Performance_Level']
    
    with col1:
        st.metric("Employee ID", f"{employee_row['Employee_ID']}")
    with col2:
        st.metric("Department", f"{employee_row['Department']}")
    with col3:
        st.metric("Attrition Probability", f"{risk_pct:.1f}%", delta=risk_level, delta_color="inverse")
    with col4:
        st.metric("Performance Forecast", f"{employee_row['Forecasted_KPI_Score']:.1f}/120", delta=perf_level)

def render_employee_search(insights_df):
    """
    orchestrates the search workflow and displays a deep dive into an individual workforce record.
    """
    st.markdown("### Individual Employee Analytics")
    st.markdown("---")
    
    #create an auto-complete search bar mapping all available employee ids
    employee_ids = insights_df['Employee_ID'].tolist()
    selected_id = st.selectbox("Select or Type Employee ID to Review:", options=employee_ids)
    
    if selected_id:
        #isolate the row corresponding to the selected employee id
        employee_row = insights_df[insights_df['Employee_ID'] == selected_id].iloc[0]
        
        #render top level metric profile header
        render_employee_profile_card(employee_row)
        st.markdown("---")
        
        #display detailed analytical breakdowns using clean column structures
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.markdown("##### Workforce Behavioral Metrics")
            metrics_data = {
                "Metric Metric Name": [
                    "Satisfaction Score (1-5)",
                    "Compa-Ratio (Salary vs Market)",
                    "Total Overtime Hours Logged",
                    "Training Hours Completed"
                ],
                "Observed Value": [
                    f"{employee_row['Satisfaction_Score']}",
                    f"{employee_row['Compa_Ratio']:.2f}",
                    f"{employee_row['Total_Overtime_Hours']:.1f} hrs",
                    f"{employee_row['Training_Hours_Completed']} hrs"
                ]
            }
            st.table(pd.DataFrame(metrics_data))
            
        with right_col:
            st.markdown("##### Operational Anomaly & Predictive Flags")
            anomaly_data = {
                "Risk Variable": [
                    "Attendance Anomaly Status",
                    "Critical Flight Risk Trigger",
                    "9-Box Grid Quadrant Assignment"
                ],
                "Current Status Flag": [
                    f"{employee_row['Attendance_Status']}",
                    "TRIGGERED" if employee_row['Critical_Flight_Risk'] == 1 else "NOMINAL",
                    f"{employee_row['Flight_Risk_Level']} Risk - {employee_row['Performance_Level']} Performance"
                ]
            }
            st.table(pd.DataFrame(anomaly_data))
            
        #save selected employee details to session state for seamless consumption by the ai engine tab
        st.session_state['selected_employee'] = employee_row.to_dict()

def render_ai_strategy(insights_df, dept_df, engine):
    """
    renders the generative ai interface for executing the recommendation engine.
    """
    st.markdown("### AI Strategy & Intervention Engine")
    st.markdown("---")
    
    # section 1: individual employee interventions
    st.markdown("#### Individual Employee Interventions")
    
    # pull the employee list
    employee_ids = insights_df['Employee_ID'].tolist()
    
    # intelligently default the dropdown to the employee they searched for in the previous tab
    default_index = 0
    if 'selected_employee' in st.session_state:
        emp_id = st.session_state['selected_employee']['Employee_ID']
        if emp_id in employee_ids:
            default_index = employee_ids.index(emp_id)
            
    selected_id = st.selectbox("Select Employee for AI Analysis:", options=employee_ids, index=default_index)
    
    if selected_id:
        employee_dict = insights_df[insights_df['Employee_ID'] == selected_id].iloc[0].to_dict()
        
        # layout buttons side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Retention Strategy", use_container_width=True):
                with st.spinner("analyzing risk factors and generating strategy via groq..."):
                    strategy = engine.get_retention_strategy(employee_dict)
                    st.success("Retention Strategy Generated")
                    st.info(strategy)
                    
        with col2:
            if st.button("Generate Upskilling Plan", use_container_width=True):
                with st.spinner("analyzing skill gaps and generating training plan via groq..."):
                    plan = engine.get_upskilling_recommendation(employee_dict)
                    st.success("Upskilling Plan Generated")
                    st.info(plan)
                    
    st.markdown("---")
    
    # section 2: macro workforce optimization
    st.markdown("#### Macro Workforce Optimization")
    st.write("Generate a macroscopic organizational brief based on cross-departmental attrition and performance metrics.")
    
    if st.button("Run Global Workforce Analysis", type="primary", use_container_width=True):
        with st.spinner("compiling departmental metrics and querying executive strategy..."):
            macro_strategy = engine.get_workforce_optimization(dept_df)
            st.success("Global Analysis Complete")
            st.markdown(f"> {macro_strategy}")