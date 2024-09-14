# run.py

from data_stream import simulate_data_stream
from anomaly_detection import AdaptiveAnomalyDetector
from visualizer import RealTimeVisualizer

def main():
    # Initialize anomaly detector and visualizer
    anomaly_detector = AdaptiveAnomalyDetector(window_size=100)
    visualizer = RealTimeVisualizer()

    # Stream and process data
    for data_point in simulate_data_stream():
        anomalies_detected = anomaly_detector.detect_anomalies(data_point)
        
        # Check if any anomaly was detected by any method
        if anomalies_detected:
            print(f"Anomaly detected by: {', '.join(anomalies_detected)}")
        
        # Update the visualization: mark anomalies
        visualizer.update_plot(data_point, anomalies_detected)

if __name__ == "__main__":
    main()
