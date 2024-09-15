import matplotlib.pyplot as plt
import numpy as np

class RealTimeVisualizer:
    def __init__(self):
        self.data_points = []
        self.anomalies = []
        self.anomaly_scores = []

        # Set up interactive plotting
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.ax1.set_title('Real-Time Data Stream')
        self.ax2.set_title('Anomaly Scores')
        self.ax2.set_xlabel('Time')

        # Define markers and colors for different anomaly detection methods
        self.marker_styles = {
            'EMA': {'marker': '^', 'color': 'green'},
            'Isolation Forest': {'marker': 's', 'color': 'blue'},
            'Seasonal': {'marker': '*', 'color': 'black'}
        }

        # Create legend handles
        self.legend_handles = []
        for method, style in self.marker_styles.items():
            handle = plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=10, linestyle='None')
            self.legend_handles.append(handle)

    def update_plot(self, data_point, anomalies_detected, anomaly_scores=None):
        self.data_points.append(data_point)
        if anomaly_scores is not None:
            if isinstance(anomaly_scores, (list, np.ndarray)):
                anomaly_scores = [float(score) for score in anomaly_scores if isinstance(score, (int, float))]
            self.anomaly_scores.append(np.mean(anomaly_scores) if anomaly_scores else 0)
        else:
            self.anomaly_scores.append(0)

        if anomalies_detected:
            self.anomalies.append(anomalies_detected)
        else:
            self.anomalies.append(None)

        # Clear and update the first axis (data stream)
        self.ax1.clear()
        self.ax1.plot(self.data_points, label="Data Stream", color='grey')

        for i, anomaly in enumerate(self.anomalies):
            if anomaly is not None:
                for method in anomaly:
                    style = self.marker_styles.get(method, {'marker': 'x', 'color': 'black'})
                    self.ax1.scatter(i, self.data_points[i], marker=style['marker'], color=style['color'], label=method)

        self.ax1.set_ylabel('Value')
        self.ax1.legend(handles=self.legend_handles, labels=self.marker_styles.keys())

        # Clear and update the second axis (anomaly scores)
        self.ax2.clear()
        self.ax2.plot(self.anomaly_scores, label="Anomaly Score", color='red', linestyle='--')

        if self.anomaly_scores:
            threshold = np.mean(self.anomaly_scores) + 3 * np.std(self.anomaly_scores)  # Dynamic threshold
            self.ax2.axhline(y=threshold, color='purple', linestyle='--', label='Anomaly Threshold')

        self.ax2.set_ylabel('Score')
        self.ax2.legend(loc='upper right')

        # Update layout and redraw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

