# anomaly_detection.py

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

class AdaptiveAnomalyDetector:
    def __init__(self, window_size=100, alpha=0.3, contamination=0.01, season_length=50):
        """
        Initialize the adaptive anomaly detector with:
        - window_size: Size of sliding window for storing recent data points.
        - alpha: Smoothing factor for Exponential Moving Average (EMA).
        - contamination: The proportion of anomalies expected, used by the Isolation Forest.
        - season_length: Number of data points to account for one complete seasonal cycle.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.season_length = season_length

        # For Isolation Forest and EMA methods
        self.data_window = deque(maxlen=window_size)
        self.ema_window = deque(maxlen=window_size)

        # Isolation Forest model
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.forest_fitted = False

    def update_ema(self, data_point):
        """
        Update the Exponential Moving Average (EMA) based on the new data point.
        """
        if len(self.ema_window) == 0:
            ema_value = data_point  # Initialize EMA with the first value
        else:
            ema_value = (self.alpha * data_point) + ((1 - self.alpha) * self.ema_window[-1])
        
        self.ema_window.append(ema_value)
        return ema_value

    def detect_anomaly_ema(self, data_point):
        """
        EMA-based anomaly detection: Check if the data point deviates significantly from the EMA.
        """
        ema_value = self.update_ema(data_point)

        if len(self.ema_window) < self.window_size:
            return False  # Wait until we have enough data points

        # Detect anomalies based on 3 standard deviations from EMA
        threshold = np.std(self.ema_window) * 3
        return abs(data_point - ema_value) > threshold

    def detect_anomaly_isolation_forest(self, data_point):
        """
        Isolation Forest-based anomaly detection: Detect if the point is anomalous using Isolation Forest.
        """
        self.data_window.append(data_point)

        # Train Isolation Forest once we have enough data
        if len(self.data_window) == self.window_size:
            if not self.forest_fitted:
                self.isolation_forest.fit(np.array(self.data_window).reshape(-1, 1))
                self.forest_fitted = True

        if self.forest_fitted:
            prediction = self.isolation_forest.predict([[data_point]])
            return prediction == -1  # -1 means anomaly

        return False

    def detect_anomaly_seasonality(self, data_point):
        """
        Seasonal-based anomaly detection: Check if data point deviates from expected seasonal pattern.
        """
        seasonal_index = len(self.data_window) % self.season_length

        if len(self.data_window) < self.season_length:
            return False  # Not enough data to detect seasonal anomalies

        # Compare the data point with the average for the corresponding seasonal index
        season_average = np.mean([self.data_window[i] for i in range(seasonal_index, len(self.data_window), self.season_length)])
        return abs(data_point - season_average) > (3 * np.std(self.data_window))

    def detect_anomalies(self, data_point):
        """
        Detect anomalies using EMA, Isolation Forest, and seasonal variation detection.
        Returns a list of methods that detected the anomaly.
        """
        anomalies_detected = []

        # EMA-based anomaly detection
        if self.detect_anomaly_ema(data_point):
            anomalies_detected.append('EMA')

        # Isolation Forest-based anomaly detection
        if self.detect_anomaly_isolation_forest(data_point):
            anomalies_detected.append('Isolation Forest')

        # Seasonality-based anomaly detection
        if self.detect_anomaly_seasonality(data_point):
            anomalies_detected.append('Seasonal')

        return anomalies_detected  # Return a list of detected anomaly types
