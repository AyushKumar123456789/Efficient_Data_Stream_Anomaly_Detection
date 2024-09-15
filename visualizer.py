import matplotlib.pyplot as plt
import numpy as np

class RealTimeVisualizer:
    def __init__(self):
        self.data_points = []
        self.anomalies = []
        self.anomaly_scores = []
        self.seasonal_components = []
        self.trend_components = []

        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        self.ax1.set_title('Real-Time Data Stream')
        self.ax2.set_title('Anomaly Scores')
        self.ax3.set_title('Seasonality and Trend')
        self.ax3.set_xlabel('Time')

        self.marker_styles = {
            'EMA': {'marker': '^', 'color': 'green'},
            'Isolation Forest': {'marker': 's', 'color': 'blue'},
            'Seasonal': {'marker': '*', 'color': 'black'}
        }

        self.legend_handles = []
        for method, style in self.marker_styles.items():
            handle = plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=10, linestyle='None')
            self.legend_handles.append(handle)

    def update_plot(self, data_point, anomalies_detected, anomaly_scores=None):
        self.data_points.append(data_point)
        self.anomaly_scores.append(np.mean(anomaly_scores) if anomaly_scores else 0)
        self.anomalies.append(anomalies_detected)

        # Simulate extraction of trend and seasonal components for demonstration
        if len(self.data_points) > 50:
            trend_component = np.polyfit(range(len(self.data_points)), self.data_points, 1)[0] * np.arange(len(self.data_points))
            seasonal_component = np.array(self.data_points) - trend_component
            self.trend_components.append(trend_component[-1])
            self.seasonal_components.append(np.mean(seasonal_component))

        # Update the data stream plot
        self.ax1.clear()
        self.ax1.plot(self.data_points, label="Data Stream", color='grey')

        for i, anomaly in enumerate(self.anomalies):
            if anomaly is not None:
                for method in anomaly:
                    style = self.marker_styles.get(method, {'marker': 'x', 'color': 'black'})
                    self.ax1.scatter(i, self.data_points[i], marker=style['marker'], color=style['color'], label=method)

        self.ax1.set_ylabel('Value')
        self.ax1.legend(handles=self.legend_handles, labels=self.marker_styles.keys())

        # Update the anomaly scores plot
        self.ax2.clear()
        self.ax2.plot(self.anomaly_scores, label="Anomaly Score", color='red', linestyle='--')

        if self.anomaly_scores:
            threshold = np.mean(self.anomaly_scores) + 3 * np.std(self.anomaly_scores)
            self.ax2.axhline(y=threshold, color='purple', linestyle='--', label='Anomaly Threshold')

        self.ax2.set_ylabel('Score')
        self.ax2.legend(loc='upper right')

        # Update the seasonality and trend plot
        self.ax3.clear()
        if self.trend_components:
            self.ax3.plot(self.trend_components, label="Trend Component", color='orange')
        if self.seasonal_components:
            self.ax3.plot(self.seasonal_components, label="Seasonal Component", color='cyan')

        self.ax3.set_ylabel('Component Value')
        self.ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
