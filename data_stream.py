# data_stream.py

import numpy as np
import time

def simulate_data_stream(frequency=60, noise_level=2, amplitude=10, anomaly_chance=0.01, trend_rate=0.1):
    """
    Simulates a continuous data stream with seasonal variations, regular pattern, noise, and occasional anomalies.
    
    Parameters:
    - frequency (int): Period of the seasonal component in seconds (default: 60s for a 1-minute period).
    - noise_level (float): Standard deviation of the normal distribution for the noise (default: 2).
    - amplitude (float): Amplitude of the sinusoidal seasonal component (default: 10).
    - anomaly_chance (float): Probability of an anomalous spike occurring (default: 1%).
    - trend_rate (float): Rate of regular pattern increase or decrease (default: 0.1 for a slow linear increase).
    
    Returns:
    A generator yielding floating-point numbers.
    """
    start_time = time.time()  # Starting point for regular pattern (linear trend)
    
    while True:
        current_time = time.time()
        
        # Regular pattern (linear increase or decrease over time)
        regular_pattern = trend_rate * (current_time - start_time)
        
        # Generate the seasonal component (sinusoidal pattern)
        seasonal = amplitude * np.sin(2 * np.pi * (current_time % frequency) / frequency)
        
        # Add random noise
        noise = np.random.normal(0, noise_level)
        
        # Combine regular pattern, seasonal, and noise to create the data point
        data_point = regular_pattern + seasonal + noise
        
        # Introduce rare anomalous spikes
        if np.random.rand() < anomaly_chance:
            data_point *= 10  # Anomalous spike by scaling the value
        
        yield data_point
        
        # Simulate the data stream's time interval
        time.sleep(0.1)  # Generates roughly 10 data points per second

# Example usage
if __name__ == "__main__":
    data_stream = simulate_data_stream()

    for _ in range(100):  # Fetch 100 data points for demonstration
        print(next(data_stream))
