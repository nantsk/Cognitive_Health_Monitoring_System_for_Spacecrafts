#Fault Remedial Actions
import pandas as pd
import plotly.graph_objects as go
import random

# Load the anomalies_detected.csv file
def load_anomalies(anomalies_csv):
    return pd.read_csv(anomalies_csv)

# Function to analyze and diagnose the technical cause of the anomaly
def diagnose_anomaly(row):
    # Initialize the cause, repair, and anomaly type
    cause = None
    repair = None
    anomaly_type = None

    # Check if the row has an anomaly
    if row['Anomaly']:
        # Point Anomalies
        if row['Point_Anomaly']:
            anomaly_type = "Point Anomaly"
            cause_options = [
                "Sudden spike/drop in readings.",
                "External disturbances (vibrations, electrical interference).",
                "Sensor noise or random spikes.",
                "Human error during manual input or calibration.",
                "Data transmission errors.",
                "System overload or communication delays."
            ]
            repair_options = [
                "Check sensor connections for loose wiring or malfunction.",
                "Inspect the environment for temporary disturbances or interferences.",
                "Apply data smoothing techniques or filtering to reduce noise.",
                "Recalibrate sensors and review manual processes for accuracy.",
                "Verify data integrity during transmission, and check for packet loss.",
                "Optimize system load handling and improve data transmission reliability."
            ]
            cause = random.choice(cause_options)
            repair = random.choice(repair_options)

        # Contextual Anomalies
        elif row['Contextual_Anomaly']:
            anomaly_type = "Contextual Anomaly"
            cause_options = [
                "Behavior abnormal compared to surrounding data.",
                "Contextual dependencies (time, location, operational phase).",
                "Unexpected environmental changes (temperature, humidity).",
                "Mismatch between current and historical context.",
                "Dynamic system behavior changes.",
                "Incorrect calibration for the operating environment."
            ]
            repair_options = [
                "Investigate if the anomaly is context-specific, review operating conditions.",
                "Ensure systems are calibrated according to their current operational context.",
                "Monitor and control environmental variables affecting sensor performance.",
                "Update system calibration and comparison models to reflect current contexts.",
                "Ensure smooth transitions between different operational modes.",
                "Recalibrate the system to match the current operational parameters."
            ]
            cause = random.choice(cause_options)
            repair = random.choice(repair_options)

        # Collective Anomalies
        elif row['Collective_Anomaly']:
            anomaly_type = "Collective Anomaly"
            cause_options = [
                "Persistent issues across multiple readings.",
                "Gradual degradation in hardware (wear and tear).",
                "Cascading failures in interconnected components.",
                "Emerging trends or sustained abnormal behavior.",
                "Faulty algorithms or misconfigured systems.",
                "External long-term trends (e.g., temperature, system aging).",
                "Data drift or out-of-distribution events.",
                "Software bugs or memory leaks."
            ]
            repair_options = [
                "Investigate hardware faults, and check multiple sensors for consistent performance.",
                "Conduct preventive maintenance and replace aging components.",
                "Analyze system dependencies and fix issues at the root cause.",
                "Identify and address the underlying long-term trend causing collective anomalies.",
                "Review and fix the algorithms responsible for anomaly detection.",
                "Adjust system parameters to handle long-term external environmental changes.",
                "Update models or recalibrate the system to handle new data distributions.",
                "Regularly update software, fix memory leaks, and enhance system reliability."
            ]
            cause = random.choice(cause_options)
            repair = random.choice(repair_options)

        # High Anomaly Score (Fallback)
        elif row['Anomaly_Score'] > 5:
            anomaly_type = "High Anomaly Score"
            cause_options = [
                "High anomaly score suggests a critical malfunction in the sensor or system.",
                "Severe error detected due to extreme deviation in sensor readings.",
                "High-level anomaly caused by potential sensor failure or external disturbance."
            ]
            repair_options = [
                "Recalibrate sensors to ensure they are properly aligned with system specifications.",
                "Inspect sensors for physical damage or obstructions that could be affecting readings.",
                "Check for external influences such as power fluctuations, electromagnetic interference, or environmental changes."
            ]
            cause = random.choice(cause_options)
            repair = random.choice(repair_options)

        # Fallback for other types of anomalies
        else:
            anomaly_type = "Unknown Anomaly"
            cause_options = [
                "An anomaly occurred, but the specific type is unclear.",
                "Anomalous behavior detected without a clear cause.",
                "System anomaly detected, requiring further inspection."
            ]
            repair_options = [
                "Perform a detailed inspection of all sensors and hardware components for potential issues.",
                "Run diagnostics on the system to identify any subtle problems that may not be immediately apparent.",
                "Increase monitoring frequency to gather more data for better analysis."
            ]
            cause = random.choice(cause_options)
            repair = random.choice(repair_options)
    
    else:
        anomaly_type = "No Anomaly"
        cause = "No anomaly detected."
        repair = "System functioning normally."

    return anomaly_type, cause, repair

# Function to analyze the anomalies and display suggestions in a Plotly table
def analyze_anomalies(anomalies_df):
    anomalies_filtered = anomalies_df[anomalies_df['Anomaly'] == True]

    if anomalies_filtered.empty:
        print("No anomalies detected.")
        return

    # Prepare data for the table
    table_data = {'Time': [], 'Anomaly Score': [], 'Anomaly Type': [], 'Cause': [], 'Suggested Repair': []}
    for index, row in anomalies_filtered.iterrows():
        anomaly_type, cause, repair = diagnose_anomaly(row)
        table_data['Time'].append(row['Time'])
        table_data['Anomaly Score'].append(row['Anomaly_Score'])
        table_data['Anomaly Type'].append(anomaly_type)
        table_data['Cause'].append(cause)
        table_data['Suggested Repair'].append(repair)

    # Create and display a Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Time", "Anomaly Score", "Anomaly Type", "Cause", "Suggested Repair"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data['Time'], table_data['Anomaly Score'], table_data['Anomaly Type'], 
                           table_data['Cause'], table_data['Suggested Repair']],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Anomaly Analysis Results", height=600, width=800)
    fig.show()

# Main function to run the analysis and display results
def analyze_and_display_repairs(anomalies_csv):
    anomalies_df = load_anomalies(anomalies_csv)
    analyze_anomalies(anomalies_df)

# Run the analysis on anomalies_detected.csv
if __name__ == "__main__":
    analyze_and_display_repairs('anomalies_detected.csv')
