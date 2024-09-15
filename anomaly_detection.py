import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AdaptiveAnomalyDetector:
    def __init__(self, window_size=100, alpha=0.3, contamination=0.01, season_length=50, drift_detection_interval=200):
        self.window_size = window_size
        self.alpha = alpha
        self.season_length = season_length
        self.drift_detection_interval = drift_detection_interval
        self.data_window = deque(maxlen=window_size)
        self.ema_window = deque(maxlen=window_size)
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.forest_fitted = False
        self.concept_drift_counter = 0
        self.scaler = StandardScaler()

    def update_ema(self, data_point):
        if len(self.ema_window) == 0:
            ema_value = data_point
        else:
            ema_value = (self.alpha * data_point) + ((1 - self.alpha) * self.ema_window[-1])
        self.ema_window.append(ema_value)
        return ema_value

    def detect_anomaly_ema(self, data_point):
        ema_value = self.update_ema(data_point)
        if len(self.ema_window) < self.window_size:
            return False, 0
        ema_window_list = list(self.ema_window)
        threshold = np.std(ema_window_list[-self.season_length:]) * 2
        score = abs(data_point - ema_value) / threshold
        return abs(data_point - ema_value) > threshold, score

    def detect_anomaly_isolation_forest(self, data_point):
        self.data_window.append(data_point)
        self.concept_drift_counter += 1
        if len(self.data_window) == self.window_size:
            if not self.forest_fitted or self.concept_drift_counter >= self.drift_detection_interval:
                data_array = np.array(self.data_window).reshape(-1, 1)
                self.scaler.fit(data_array)
                scaled_data = self.scaler.transform(data_array)
                self.isolation_forest.fit(scaled_data)
                self.forest_fitted = True
                self.concept_drift_counter = 0
        if self.forest_fitted:
            scaled_point = self.scaler.transform([[data_point]])
            score = -self.isolation_forest.decision_function(scaled_point)[0]
            prediction = self.isolation_forest.predict(scaled_point)
            return prediction == -1, score
        return False, 0


    def detect_anomaly_seasonality(self, data_point):
        if len(self.data_window) < self.season_length:
            return False, 0
        seasonal_index = len(self.data_window) % self.season_length
        season_average = np.mean([self.data_window[i] for i in range(seasonal_index, len(self.data_window), self.season_length)])
        score = abs(data_point - season_average) / (3 * np.std(self.data_window))
        return abs(data_point - season_average) > (3 * np.std(self.data_window)), score

    def detect_anomalies(self, data_point):
        anomalies_detected = []
        scores = []
        anomaly, score = self.detect_anomaly_ema(data_point)
        if anomaly:
            anomalies_detected.append('EMA')
            scores.append(score)

        anomaly, score = self.detect_anomaly_isolation_forest(data_point)
        if anomaly:
            anomalies_detected.append('Isolation Forest')
            scores.append(score)

        anomaly, score = self.detect_anomaly_seasonality(data_point)
        if anomaly:
            anomalies_detected.append('Seasonal')
            scores.append(score)

        return anomalies_detected, scores
