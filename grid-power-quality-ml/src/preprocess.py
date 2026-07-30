import pandas as pd
import numpy as np
#Creating Synthetic Frequency
def preprocessing(df):
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],dayfirst=True)
    df = df.set_index("Datetime")
    df = df.drop(columns=['Date', 'Time'])
    df = df.interpolate(method='time', limit=10, numeric_only=True)
    #rows_with_nan = df[df.isnull().any(axis=1)]
    f_nominal = 50
    noise = np.random.normal(0, 0.02, size=(len(df)) )
    power_normalized = (df['Global_active_power'] - df['Global_active_power'].mean()) / df['Global_active_power'].std()
    load_effect = -0.015 * power_normalized
    df['Frequency'] = f_nominal + noise + load_effect
    df['Frequency'] = df['Frequency'].round(2)
    return df 