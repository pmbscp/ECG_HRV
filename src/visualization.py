import matplotlib.pyplot as plt

def multi_ecg_visu(ecg_segments, polarities, segment_labels=None, title="ECG Segments", xlabel="Time (ms)", ylabel="Amplitude", figsize=(15, 10)):
    """
    Plots multiple ECG segments on the same graph.

    Parameters:
    - ecg_segments: List of DataFrames, where each DataFrame is a segment of ECG data.
    - segment_labels: List of labels for each ECG segment (optional).
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Size of the figure.
    - polarity: If True, invert the ECG signal.
    """
    plt.figure(figsize=figsize)

    for i, segment in enumerate(ecg_segments):
        label = segment_labels[i] if segment_labels and i < len(segment_labels) else f"Segment {i+1}"
        y_data = -segment['EcgWaveform'] if polarities[i] else segment['EcgWaveform']
        plt.plot(segment['timestamp_ms'], y_data, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()