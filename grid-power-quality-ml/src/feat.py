import numpy as np
import pandas as pd
def train_features(df,voltage_nominal=241,frequency_nominal=50,rolling_window=10,):
    df = df.copy()
    # Apparent Power (VA)
    df["apparent_power"] = np.sqrt(
        df["Global_active_power"]**2 +
        df["Global_reactive_power"]**2
    )

    # Power Factor
    df["power_factor"] = np.where(
        df["apparent_power"] != 0,
        df["Global_active_power"] / df["apparent_power"],
        0
    )

    # Voltage Deviation
    df["voltage_deviation"] = (
        df["Voltage"] - voltage_nominal
    )

    # Frequency Deviation
    df["frequency_deviation"] = (
        df["Frequency"] - frequency_nominal
    )

    df["voltage_mean"] = (
        df["Voltage"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    df["frequency_mean"] = (
        df["Frequency"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    df["current_mean"] = (
        df["Global_intensity"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    df["active_power_mean"] = (
        df["Global_active_power"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    df["reactive_power_mean"] = (
        df["Global_reactive_power"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )


    df["voltage_std"] = (
        df["Voltage"]
        .rolling(rolling_window, min_periods=1)
        .std()
    )

    df["frequency_std"] = (
        df["Frequency"]
        .rolling(rolling_window, min_periods=1)
        .std()
    )

    df["current_std"] = (
        df["Global_intensity"]
        .rolling(rolling_window, min_periods=1)
        .std()
    )

    df["active_power_std"] = (
        df["Global_active_power"]
        .rolling(rolling_window, min_periods=1)
        .std()
    )

    df["reactive_power_std"] = (
        df["Global_reactive_power"]
        .rolling(rolling_window, min_periods=1)
        .std()
    )


    df["voltage_change"] = df["Voltage"].diff()

    df["frequency_change"] = df["Frequency"].diff()

    df["current_change"] = df["Global_intensity"].diff()

    df["active_power_change"] = (
        df["Global_active_power"].diff()
    )

    df["reactive_power_change"] = (
        df["Global_reactive_power"].diff()
    )


    df["hour"] = df.index.hour

    df["day_of_week"] = (
        df.index.dayofweek
    )

    df["month"] = df.index.month

    df["weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    #df.drop(columns=["hour", "day_of_week", "month"], inplace=True, errors='ignore')
    df = df.dropna()
    return df