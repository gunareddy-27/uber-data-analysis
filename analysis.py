import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk

def analyze_data(df):
    """Analyze Uber ride data and return trip distributions."""
    
    if df is None or df.empty:
        print("⚠ WARNING: DataFrame is empty. Returning empty analysis results.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    if "START_DATE" not in df.columns:
        print("⚠ ERROR: 'START_DATE' column missing! Cannot analyze time-based data.")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # Convert to datetime
    df["START_DATE"] = pd.to_datetime(df["START_DATE"], errors='coerce')

    # Hourly trip distribution
    df["Hour"] = df["START_DATE"].dt.hour
    hourly_trips = df["Hour"].value_counts().sort_index()

    # Daily trip distribution
    daily_trips = df["START_DATE"].dt.date.value_counts().sort_index()

    # Weekday trip distribution
    df["Weekday"] = df["START_DATE"].dt.day_name()
    weekday_trips = df["Weekday"].value_counts()

    return hourly_trips, daily_trips, weekday_trips



if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('C:\\Users\\anush\\OneDrive\\Desktop\\uber\\datasets\\UberDataset.csv')

    df['pickup_datetime'] = pd.to_datetime(df['START_DATE'])

    # Run analysis
    hourly_trips, daily_trips, weekday_trips = analyze_data(df)

    # Create a Tkinter window
    window = tk.Tk()
    window.title("Uber Dataset Analysis")

    # Create a dropdown menu
    selected_plot = tk.StringVar()
    selected_plot.set("Hourly Trips")
    options = ["Hourly Trips", "Daily Trips", "Weekday Trips"]
    dropdown = ttk.OptionMenu(window, selected_plot, *options)
    dropdown.pack()

    # Create a figure and axis
    figure = Figure(figsize=(8, 6), dpi=100)
    axis = figure.add_subplot(111)

    # Create a canvas to display the plot
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Function to update the plot
    def update_plot():
        axis.clear()
        if selected_plot.get() == "Hourly Trips":
            axis.plot(hourly_trips)
        elif selected_plot.get() == "Daily Trips":
            axis.plot(daily_trips)
        elif selected_plot.get() == "Weekday Trips":
            axis.plot(weekday_trips)
        canvas.draw()

    # Button to update the plot
    button = tk.Button(master=window, text="Update Plot", command=update_plot)
    button.pack()

    # Start the Tkinter event loop
    window.mainloop()