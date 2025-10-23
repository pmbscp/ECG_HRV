import pandas as pd
import neurokit2 as nk

def segmentation_ecg(ecg_df, log_event_df):
    """
    Segments the ECG data based on the event timestamps.

    Args:
        ecg_df (pd.DataFrame): DataFrame containing the ECG data.
        log_event_df (pd.DataFrame): DataFrame containing the event timestamps.

    Returns:
        dict: Dictionary of ECG DataFrames segmented per experiment's events.
    """
    segments = {}
    events = log_event_df['events'].unique()
    
    order_of_conditions = []
    condition = ''
    condition_prefixes = ['C', '0B', '2B']

    for event in events:
        if event.endswith('_begin'):
            event_name = event[:-6]
            if len(event_name.split('_')[0]) <= 2:
                condition = event_name.split('_')[0]
            if condition in condition_prefixes and condition not in order_of_conditions:
                order_of_conditions.append(condition)

            begin_time = log_event_df[log_event_df['events'] == event]['timestamp_ms'].values[0]
            end_event = event_name + '_end'
            
            if end_event in log_event_df['events'].values:
                end_time = log_event_df[log_event_df['events'] == end_event]['timestamp_ms'].values[0]
                segment = ecg_df[(ecg_df['timestamp_ms'] >= begin_time) & (ecg_df['timestamp_ms'] <= end_time)]
                segments[event_name] = segment
            elif end_event not in log_event_df['events'].values:
                name_condition = end_event.split('_')
                end_condition = name_condition[0] + '_end'
                end_time = log_event_df[log_event_df['events'] == end_condition]['timestamp_ms'].values[0]
                segment = ecg_df[(ecg_df['timestamp_ms'] >= begin_time) & (ecg_df['timestamp_ms'] <= end_time)]
                segments[event_name] = segment
                
                if event_name == 'fixation_cross':
                    begin_time_v2 = begin_time + 30000  # +30 seconds
                    end_time_v2 = end_time - 30000  # -30 seconds
                    segment_v2 = ecg_df[(ecg_df['timestamp_ms'] >= begin_time_v2) & (ecg_df['timestamp_ms'] <= end_time_v2)]
                    segments[event_name + '_v2'] = segment_v2

    def combine_phases(prefix, start_phase, end_phase):
        combined_segment = pd.DataFrame()
        for phase in range(start_phase, end_phase + 1):
            phase_name = f"{prefix}_phase_{phase}"
            if phase_name in segments:
                combined_segment = pd.concat([combined_segment, segments[phase_name]])
        return combined_segment

    for condition in order_of_conditions:
        segments[f'{condition}.1'] = combine_phases(condition, 1, 6)
        segments[f'{condition}.2'] = combine_phases(condition, 7, 12)

    return segments


def measure_segment_duration(segment):
    """
    Fonction measuring ECG segment length in seconds

    Args:
        segment (pd.DataFrame): Data segment from the ECG raw signal

    Returns:
        tuple: Duration of the segment in seconds
    """
    start_time = segment['timestamp_ms'].iloc[0]
    end_time = segment['timestamp_ms'].iloc[-1]
    duration_ms = end_time - start_time
    duration_seconds = duration_ms / 1000  # Conversion en secondes
    return duration_seconds

def clean_segment(segment, sampling_rate, method):
    """
    Cleans an ECG segment using specified method
    
    Args:
        segment (DataFrame): ECG segment to clean
        sampling_rate (int): Sampling rate of the ECG signal
        method (str): Cleaning method to use.
    
    Returns:
        DataFrame: Clean ECG segment.
    """
    methods_dict = {
        'neurokit': nk.ecg_clean,
        'pantompkins1985': lambda sig, sr: nk.ecg_clean(sig, method="pantompkins1985", sampling_rate=sr),
        'hamilton2002': lambda sig, sr: nk.ecg_clean(sig, method="hamilton2002", sampling_rate=sr),
        'elgendi2010': lambda sig, sr: nk.ecg_clean(sig, method="elgendi2010", sampling_rate=sr),
        'engzeemod2012': lambda sig, sr: nk.ecg_clean(sig, method="engzeemod2012", sampling_rate=sr),
        'vg': lambda sig, sr: nk.ecg_clean(sig, method="vg", sampling_rate=sr),
        'biosppy': lambda sig, sr: nk.ecg_clean(sig, method="biosppy", sampling_rate=sr)
    }

    if method not in methods_dict:
        raise ValueError(f"Method {method} is not supported.")
    
    cleaned_segment = segment.copy()
    cleaned_segment['EcgWaveform'] = nk.signal_sanitize(cleaned_segment['EcgWaveform'])
    cleaned_segment['EcgWaveform'] = methods_dict[method](segment['EcgWaveform'], sampling_rate)
    rpeaks = nk.ecg_peaks(cleaned_segment['EcgWaveform'], sampling_rate, method='neurokit', correct_artifacts=True)
    
    return cleaned_segment, rpeaks

def remove_short_segments(all_segments, cleaned_segments, min_length=1000, verbose=False):
    """
    Removes unnecessary or unused segments for the analysis. Too shorts segments can indeed prevent some fonctions, 
    packages or processes to work, with hrv metrics extractions among them.
    
    Args:
        DataFrame: A dataframe containing the rest of all uncleaned segments, free from too shorts segments
        DataFrame: A dataframe containing the rest of all uncleaned segments, free from too shorts segments
        min_length (int): the minimum length of segment used.
        verbose (boolean): An option to specify what is suppressed if needed.
    
    Returns:
        DataFrame: A dataframe containing the rest of all uncleaned segments, free from too shorts segments
        DataFrame: A dataframe containing the rest of all uncleaned segments, free from too shorts segments
    """    
    for participant in list(all_segments.keys()):
        for segment in list(all_segments[participant].keys()):
            ecg_signal = all_segments[participant][segment]
            if ecg_signal is None or len(ecg_signal) < min_length:
                del all_segments[participant][segment]
                if verbose == True:
                    print(f"Deleted this segment : {segment} from subject {participant}")
    
    for participant in list(cleaned_segments.keys()):
        for method in list(cleaned_segments[participant].keys()):
            for segment in list(cleaned_segments[participant][method].keys()):
                ecg_signal = cleaned_segments[participant][method][segment]
                if ecg_signal is None or len(ecg_signal) < min_length:
                    del cleaned_segments[participant][method][segment]
                    if verbose == True:
                        print(f"Deleted this segment : {segment} from subject {participant}, cleaned with : {method} method")

    return all_segments, cleaned_segments