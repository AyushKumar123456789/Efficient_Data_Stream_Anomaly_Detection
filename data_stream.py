# data_stream.py

import numpy as np
import time

def simulate_data_stream():
    """
    Simulates a continuous data stream with seasonal variations and random noise.
    Returns a generator yielding floating-point numbers.
    """
    while True:
        # Generate a periodic component (seasonality)
        seasonal = 10 * np.sin(time.time() / 60)  # Change over 1-minute period
        noise = np.random.normal(0, 2)  # Random noise component
        data_point = seasonal + noise

        # Introduce rare anomalous spikes (e.g., 1% chance of anomaly)
        if np.random.rand() < 0.01:
            data_point *= 10  # Anomalous spike

        yield data_point
        time.sleep(0.1)  # Simulate a streaming rate of 10 points per second
