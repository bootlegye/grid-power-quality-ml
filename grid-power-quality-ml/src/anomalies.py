import pandas as pd
import numpy as np

#Alternative to np.sin
"""
def create_offset_waveform(duration, peak):
    half = duration // 2
    return np.concatenate([
        np.linspace(0, peak, half),
        np.linspace(peak, 0, duration-half)
    ])
"""

def create_waveform(duration, peak):
    half = duration // 2
    return np.concatenate([
        np.linspace(1.0, peak, half),
        np.linspace(peak, 1.0, duration - half)
    ])

def voltage_sag(df, start, duration):
    df.iloc[start:start+duration, df.columns.get_loc("Voltage")] *= create_waveform(
        duration,
        np.random.uniform(0.50, 0.75))
    df.iloc[start:start+duration, df.columns.get_loc("Global_intensity")] *= create_waveform(
        duration,
        np.random.uniform(1.20, 1.50))
    df.iloc[start:start+duration, df.columns.get_loc("Global_active_power")] *= create_waveform(
        duration,
        np.random.uniform(0.40, 0.65)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_reactive_power")] *= create_waveform(
        duration,
        np.random.uniform(1.50, 1.75)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Frequency")] += np.sin(
        np.linspace(0, np.pi, duration)
    ) * np.random.uniform(-0.30, -0.10)
    df.iloc[start:start+duration, df.columns.get_loc("label")] = 1


def voltage_swell(df, start, duration):

    df.iloc[start:start+duration, df.columns.get_loc("Voltage")] *= create_waveform(
        duration,
        np.random.uniform(1.10, 1.25)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_intensity")] *= create_waveform(
        duration,
        np.random.uniform(0.55, 0.75))
    df.iloc[start:start+duration, df.columns.get_loc("Global_active_power")] *= create_waveform(
        duration,
        np.random.uniform(1.30, 1.55)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_reactive_power")] *= create_waveform(
        duration,
        np.random.uniform(0.60, 0.80))

    df.iloc[start:start+duration, df.columns.get_loc("Frequency")] += np.sin(
        np.linspace(0, np.pi, duration)
    ) * np.random.uniform(0.05, 0.20)
    df.iloc[start:start+duration, df.columns.get_loc("label")] = 1


def frequency_div(df, start, duration):

    df.iloc[start:start+duration, df.columns.get_loc("Frequency")] += np.sin(
        np.linspace(0, 2*np.pi, duration)
    ) * np.random.uniform(0.30, 0.70)

    df.iloc[start:start+duration, df.columns.get_loc("Voltage")] *= create_waveform(
        duration,
        np.random.uniform(1.15, 1.35))
    df.iloc[start:start+duration, df.columns.get_loc("Global_intensity")] *= create_waveform(
        duration,
        np.random.uniform(1.20, 1.45) )
    df.iloc[start:start+duration, df.columns.get_loc("label")] = 1


def current_surge(df, start, duration):
    df.iloc[start:start+duration, df.columns.get_loc("Global_intensity")] *= create_waveform(
        duration,
        np.random.uniform(1.50, 2.00)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_active_power")] *= create_waveform(
        duration,
        np.random.uniform(1.20, 1.60)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_reactive_power")] *= create_waveform(
        duration,
        np.random.uniform(1.70, 1.90)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Voltage")] *= create_waveform(
        duration,
        np.random.uniform(0.94, 0.98)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Frequency")] += np.sin(
        np.linspace(0, np.pi, duration)
    ) * np.random.uniform(-0.35, -0.15)
    df.iloc[start:start+duration, df.columns.get_loc("label")] = 1


def load_spike(df, start, duration):
    df.iloc[start:start+duration, df.columns.get_loc("Global_active_power")] *= create_waveform(
        duration,
        np.random.uniform(1.50, 2.00)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_reactive_power")] *= create_waveform(
        duration,
        np.random.uniform(1.30, 1.80)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Global_intensity")] *= create_waveform(
        duration,
        np.random.uniform(1.40, 2.00)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Voltage")] *= create_waveform(
        duration,
        np.random.uniform(0.95, 0.99)
    )
    df.iloc[start:start+duration, df.columns.get_loc("Frequency")] += np.sin(
        np.linspace(0, np.pi, duration)
    ) * np.random.uniform(-0.10, -0.03)
    df.iloc[start:start+duration, df.columns.get_loc("label")] = 1


def inject_anomalies(df, num_events):

    np.random.seed(42)
    df = df.copy()
    df["label"] = 0

    event_functions = [
        voltage_sag,
        voltage_swell,
        frequency_div,
        current_surge,
        load_spike,
    ]

    for _ in range(num_events):
        duration = np.random.randint(20, 60)
        start = np.random.randint(0, len(df) - duration)
        np.random.choice(event_functions)(df, start, duration)
    return df