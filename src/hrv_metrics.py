import pandas as pd
import neurokit2 as nk

def extract_metrics_hrv(segment, method, fs=250):
    """
    Extracts HRV metrics from an ECG segment using the NeuroKit2 Python package.

    Args:
        segment (pd.DataFrame): ECG data segment.
        method (str): the method used for extraction of HRV metrics
        fs (int): Sampling frequency of the ECG signal.

    Returns:
        dict: Dictionary containing computed HRV metrics.
        list: List containing the BPM values of the detected R-peaks.
    """
    ecg_signals, info = nk.ecg_peaks(segment['EcgWaveform'], sampling_rate=fs, method=method)
    hrv_temporal_metrics = nk.hrv_time(ecg_signals, sampling_rate=fs)
    hrv_frequency_metrics = nk.hrv_frequency(ecg_signals, sampling_rate=fs)
    hrv_metrics = pd.concat([hrv_temporal_metrics, hrv_frequency_metrics], axis=1)
    HR = nk.signal_rate(ecg_signals, sampling_rate=fs)
    
    return hrv_metrics, HR


def multi_extract_hrv_metrics(cleaned_segments, segments_of_interest, cleaning_method_chosen='biosppy', cleaned=True, export=False, verbose=False):
    """
    Extracts HRV metrics from all relevant ECG segments and stores them in a new HRV metrics dictionary.


    Args:
        cleaned_segments (dict): Dictionary containing cleaned ECG segments organized by participant and method.
        segments_of_interest (list): List of segment names to include in the analysis (e.g., ['C', '0B', '2B']).
        cleaning_method_chosen (str, optional): Name of the preprocessing method to use for HRV extraction.
            Defaults to 'biosppy'.
        cleaned (bool, optional): Indicates whether the input segments have been preprocessed.
            Defaults to True.
        export (bool, optional): If True, exports the compiled HRV metrics to a CSV file named "HRV_metrics.csv".
            Defaults to False.
        verbose (bool, optional): If True, prints progress information during extraction.
            Defaults to False.
    Returns:
        dict: Nouveau dictionnaire avec les métriques HRV pour chaque segment.
    """

    hrv_metrics_dict = {}
    
    data_frames = []

    for participant, methods in cleaned_segments.items():
        hrv_metrics_dict[participant] = {}
        for method, segments in methods.items():
            if method != cleaning_method_chosen:
                continue
            hrv_metrics_dict[participant][method] = {}
            for segment_name, segment_data in segments.items():
                if segment_name in segments_of_interest:
                    if verbose:
                        print(f"Actually extracting HRV metrics of segment {segment_name} from participant {participant}")
                    hrv_metrics, HR = extract_metrics_hrv(segment_data, method='neurokit', fs=250)
                    hrv_metrics_dict[participant][method][segment_name] = hrv_metrics
                    HR_mean = sum(HR) / len(HR)
                    flat_metrics = pd.DataFrame({
                        'Participant': [participant],
                        'Segment': [segment_name],
                        'MeanHR': HR_mean
                    })
                    hrv_metrics_df = pd.DataFrame(hrv_metrics, index=[0])
                    combined_df = pd.concat([flat_metrics, hrv_metrics_df], axis=1)
                    data_frames.append(combined_df)


    df_hrv = pd.concat(data_frames, ignore_index=True)
    df_hrv.set_index(['Participant', 'Segment'], inplace=True)

    if export:
        df_hrv.to_csv("HRV_metrics.csv", index=True, sep=";")

    return hrv_metrics_dict, df_hrv