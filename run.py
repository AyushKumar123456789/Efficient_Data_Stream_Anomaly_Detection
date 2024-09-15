from data_stream import simulate_data_stream
from anomaly_detection import AdaptiveAnomalyDetector
from visualizer import RealTimeVisualizer

def main():
    anomaly_detector = AdaptiveAnomalyDetector(window_size=100)
    visualizer = RealTimeVisualizer()

    for data_point in simulate_data_stream():
        anomalies_detected, anomaly_scores = anomaly_detector.detect_anomalies(data_point)
        
        if anomalies_detected:
            print(f"Anomaly detected by: {', '.join(anomalies_detected)}")

        visualizer.update_plot(data_point, anomalies_detected, anomaly_scores)

if __name__ == "__main__":
    main()
