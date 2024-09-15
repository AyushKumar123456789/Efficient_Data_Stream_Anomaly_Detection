# anomaly_detection.py

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

class AdaptiveAnomalyDetector:
    def __init__(self, window_size=100, alpha=0.3, contamination=0.01, season_length=50, drift_detection_interval=200):
        """
        Initialize the adaptive anomaly detector with:
        - window_size: Size of sliding window for storing recent data points.
        - alpha: Smoothing factor for Exponential Moving Average (EMA).
        - contamination: The proportion of anomalies expected, used by the Isolation Forest.
        - season_length: Number of data points to account for one complete seasonal cycle.
        - drift_detection_interval: The interval at which the model checks for concept drift and retrains if necessary.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.season_length = season_length
        self.drift_detection_interval = drift_detection_interval

        # For Isolation Forest, EMA, and Concept Drift methods
        self.data_window = deque(maxlen=window_size)
        self.ema_window = deque(maxlen=window_size)

        # Isolation Forest model
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.forest_fitted = False
        self.concept_drift_counter = 0

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

        # Convert deque to list for slicing
        ema_window_list = list(self.ema_window)

        # Dynamically adjust the threshold for EMA based on recent data
        threshold = np.std(ema_window_list[-self.season_length:]) * 3
        return abs(data_point - ema_value) > threshold


    def detect_anomaly_isolation_forest(self, data_point):
        """
        Isolation Forest-based anomaly detection: Detect if the point is anomalous using Isolation Forest.
        """
        self.data_window.append(data_point)
        self.concept_drift_counter += 1

        # Train Isolation Forest once we have enough data or when drift interval is hit
        if len(self.data_window) == self.window_size:
            if not self.forest_fitted or self.concept_drift_counter >= self.drift_detection_interval:
                self.isolation_forest.fit(np.array(self.data_window).reshape(-1, 1))
                self.forest_fitted = True
                self.concept_drift_counter = 0  # Reset the drift counter

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
