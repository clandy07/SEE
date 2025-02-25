import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["DATE"] = pd.to_datetime(df["DATE"])  # Convert to datetime
    df = df.sort_values(by=["PATIENT_ID", "DATE"])  # Sort by patient and date
    return df

# Compute event intervals
def compute_event_intervals(df):
    df["prev_DATE"] = df.groupby("PATIENT_ID")["DATE"].shift(1)
    df["event_interval"] = (df["DATE"] - df["prev_DATE"]).dt.days
    df = df.dropna()
    return df

# Boxplot visualization with reference line and improved labels
def plot_boxplot(df):
    plt.figure(figsize=(12,6))
    sns.boxplot(x=df["PATIENT_ID"], y=df["event_interval"], whis=1.5)
    
    # Add horizontal reference line (mean event interval)
    mean_interval = df["event_interval"].mean()
    plt.axhline(y=mean_interval, color='red', linestyle='dashed', linewidth=1)
    
    # Adjust x-axis labels
    plt.xticks(rotation=90)
    plt.title("Distribution of Event Intervals Per Patient")
    plt.xlabel("Patient ID")
    plt.ylabel("Event Interval (Days)")
    plt.show()

# Running the pipeline
file_path = "med_events.csv"
data_df = load_data(file_path)
data_df = compute_event_intervals(data_df)
plot_boxplot(data_df)