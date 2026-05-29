import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

NUM_EMPLOYEES = 2000
START_DATE = datetime(2026, 1, 1) # 3 month of raw data
END_DATE = datetime(2026, 3, 31)

print(f"Generating raw HR datasets for {NUM_EMPLOYEES} employees...")

# ---------------------------------------------------------
# 1. EMPLOYEE MASTER DATA
# ---------------------------------------------------------
emp_ids = [f"EMP{str(i).zfill(5)}" for i in range(1, NUM_EMPLOYEES + 1)]
departments = ['Technology', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations']

master_df = pd.DataFrame({
    'Employee_ID': emp_ids,
    'Age': np.random.randint(22, 60, NUM_EMPLOYEES),
    'Department': np.random.choice(departments, NUM_EMPLOYEES, p=[0.3, 0.2, 0.05, 0.1, 0.1, 0.25]),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], NUM_EMPLOYEES, p=[0.55, 0.43, 0.02]),
    'Joining_Date': [START_DATE - timedelta(days=np.random.randint(100, 3000)) for _ in range(NUM_EMPLOYEES)]
})

# ---------------------------------------------------------
# 2. PAYROLL INFORMATION
# ---------------------------------------------------------
payroll_df = pd.DataFrame({
    'Employee_ID': emp_ids,
    'Base_Salary': np.random.randint(40000, 150000, NUM_EMPLOYEES),
    'Bonus_Last_Year': np.random.randint(0, 15000, NUM_EMPLOYEES),
    'Tax_Deductions': np.random.randint(5000, 25000, NUM_EMPLOYEES)
})

# ---------------------------------------------------------
# 3. PERFORMANCE RATINGS (Last Cycle)
# ---------------------------------------------------------
performance_df = pd.DataFrame({
    'Employee_ID': emp_ids,
    'Manager_Rating': np.random.choice([1, 2, 3, 4, 5], NUM_EMPLOYEES, p=[0.05, 0.15, 0.5, 0.2, 0.1]),
    'KPI_Achievement_Pct': np.random.randint(50, 120, NUM_EMPLOYEES),
    'Training_Hours_Completed': np.random.randint(0, 40, NUM_EMPLOYEES)
})

# ---------------------------------------------------------
# 4. EMPLOYEE ENGAGEMENT DATA
# ---------------------------------------------------------
engagement_df = pd.DataFrame({
    'Employee_ID': emp_ids,
    'Satisfaction_Score': np.random.choice([1, 2, 3, 4, 5], NUM_EMPLOYEES, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'Work_Life_Balance': np.random.choice([1, 2, 3, 4, 5], NUM_EMPLOYEES, p=[0.15, 0.25, 0.4, 0.15, 0.05])
})

# --- DETERMINE ATTRITION based on logic to make ML work later ---
# Low rating, low satisfaction, low salary = higher attrition risk
attrition_risk = (
    (performance_df['Manager_Rating'] <= 2).astype(int) * 0.3 + 
    (engagement_df['Satisfaction_Score'] <= 2).astype(int) * 0.4 +
    (payroll_df['Base_Salary'] < 60000).astype(int) * 0.2
)
# Add some randomness and assign Target Attrition
master_df['Attrition_Target'] = np.where(attrition_risk + np.random.rand(NUM_EMPLOYEES) * 0.5 > 0.6, 1, 0)

# ---------------------------------------------------------
# 5. LEAVE & SHIFT DATA
# ---------------------------------------------------------
# Generate a list of working days in the month (excluding weekends)
working_days = pd.date_range(start=START_DATE, end=END_DATE, freq='B')

leave_records = []
for emp in emp_ids:
    # Each employee randomly takes 0 to 3 days of leave in the month
    num_leaves = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
    leave_dates = np.random.choice(working_days, num_leaves, replace=False)
    for date in leave_dates:
        leave_records.append({
            'Employee_ID': emp,
            'Date': date,
            'Leave_Type': np.random.choice(['Sick', 'Casual', 'Unpaid']),
            'Shift': 'Morning' # Default shift
        })

leave_df = pd.DataFrame(leave_records)
if leave_df.empty:
    leave_df = pd.DataFrame(columns=['Employee_ID', 'Date', 'Leave_Type', 'Shift'])

# ---------------------------------------------------------
# 6. ATTENDANCE RECORDS (Raw Check-in / Check-out)
# ---------------------------------------------------------
# Create a Cartesian product of all employees and all working days
attendance_list = [{'Employee_ID': emp, 'Date': date} for emp in emp_ids for date in working_days]
attendance_df = pd.DataFrame(attendance_list)

# Left join leaves to see who was on leave
attendance_df = pd.merge(attendance_df, leave_df[['Employee_ID', 'Date', 'Leave_Type']], on=['Employee_ID', 'Date'], how='left')

# Generate raw check-in / check-out times
def generate_time(base_hour, std_dev_minutes):
    delta = timedelta(minutes=float(np.random.normal(0, std_dev_minutes)))
    base_time = datetime(2023, 1, 1, base_hour, 0, 0) + delta
    return base_time.strftime('%H:%M:%S')

# Apply times only if NOT on leave
attendance_df['Check_In'] = attendance_df['Leave_Type'].apply(
    lambda x: generate_time(9, 30) if pd.isna(x) else np.nan
)
attendance_df['Check_Out'] = attendance_df['Leave_Type'].apply(
    lambda x: generate_time(18, 45) if pd.isna(x) else np.nan
)

# Simulate some missing punch-outs (forgetting to clock out)
missing_checkout_mask = (attendance_df['Leave_Type'].isna()) & (np.random.rand(len(attendance_df)) < 0.02)
attendance_df.loc[missing_checkout_mask, 'Check_Out'] = np.nan

# Drop the temporary Leave_Type column from attendance to keep them separate
attendance_df = attendance_df.drop(columns=['Leave_Type'])

# ---------------------------------------------------------
# SAVE ALL FILES
# ---------------------------------------------------------
dir =r'D:\OFFICE\Tasks\AI_based_HR_system\data\raw'
master_df.to_csv(dir + '1_Employee_Master.csv', index=False)
payroll_df.to_csv(dir + '2_Payroll_Info.csv', index=False)
performance_df.to_csv(dir + '3_Performance_Ratings.csv', index=False)
engagement_df.to_csv(dir + '4_Engagement_Data.csv', index=False)
leave_df.to_csv(dir + '5_Leave_Shift_Data.csv', index=False)
attendance_df.to_csv(dir + '6_Attendance_Records.csv', index=False)

print(f"Success! 6 CSV files generated.")
print(f"Total Attendance rows generated: {len(attendance_df)}")