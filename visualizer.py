# visualizer.py

import matplotlib.pyplot as plt

class RealTimeVisualizer:
    def __init__(self):
        self.data_points = []
        self.anomalies = []

        # Set up interactive plotting
        plt.ion()
        self.fig, self.ax = plt.subplots()

        # Define markers and colors for different anomaly detection methods
        self.marker_styles = {
            'EMA': {'marker': '^', 'color': 'green'},  # Triangle for EMA anomalies
            'Isolation Forest': {'marker': 's', 'color': 'blue'},  # Square for Isolation Forest anomalies
            'Seasonal' : {'marker': '*', 'color': 'black'}  # Star for Seasonal anomalies
        }
        
        # Create an empty plot for legend
        self.legend_handles = []
        for method, style in self.marker_styles.items():
            handle = plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=10, linestyle='None')
            self.legend_handles.append(handle)
        
        self.ax.legend(handles=self.legend_handles, labels=self.marker_styles.keys())

    def update_plot(self, data_point, anomalies_detected):
        """
        Update the plot with the new data point and mark anomalies with different shapes.
        """
        # Store data points and mark anomalies
        self.data_points.append(data_point)

        if anomalies_detected:
            self.anomalies.append(anomalies_detected)
        else:
            self.anomalies.append(None)

        # Clear plot and redraw
        self.ax.clear()
        self.ax.plot(self.data_points, label="Data Stream")
        
        # Mark anomalies with different shapes
        for i, anomaly in enumerate(self.anomalies):
            if anomaly is not None:
                # Multiple anomalies: plot different shapes for each detection method
                for method in anomaly:
                    style = self.marker_styles.get(method, {'marker': 'x', 'color': 'black'})  # Default: X marker for unknown methods
                    self.ax.scatter(i, self.data_points[i], marker=style['marker'], color=style['color'], label=method)
        
        # Adding legend and showing plot updates
        self.ax.legend(handles=self.legend_handles, labels=self.marker_styles.keys())
        plt.pause(0.01)
