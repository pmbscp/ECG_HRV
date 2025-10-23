import os
import pandas as pd
import neurokit2 as nk

def load_data_for_participant(participant_folder):
    """
    Loads the ECG, log_event, and cognitive evaluation files for a given participant.

    Args:
        participant_folder (str): Path to the participant's folder.

    Returns:
        tuple: DataFrames for the ECG signal, experimental interface timestamps, and subjective measures.
    """
    ecg_folder = os.path.join(participant_folder, "ECG")
    log_event_path = os.path.join(participant_folder, "SIMU", "log_event.csv")
    cog_evals_path = os.path.join(participant_folder, "SIMU", "cog_evals.csv")
    participant_id = participant_folder.replace('../data/', '')
    error_path = os.path.join(participant_folder, f"Tableau_suivi_erreur_{participant_id}.csv")
    

    ecg_file = None
    if os.path.exists(ecg_folder):
        for file in os.listdir(ecg_folder):
            if 'ECG' in file and file.endswith('.csv'):
                ecg_file = os.path.join(ecg_folder, file)
                break

    if ecg_file and os.path.exists(log_event_path) and os.path.exists(cog_evals_path):
        ecg_df = pd.read_csv(ecg_file, sep=',')
        log_event_df = pd.read_csv(log_event_path, sep=';')
        cog_evals_df = pd.read_csv(cog_evals_path, sep=';')
        if os.path.exists(error_path):
            error_df = pd.read_csv(error_path, sep=';', encoding='iso-8859-1')
        else:
            print(f"missing error file from this participant : {participant_id} !!\n"
                  + "or there is some mistake going on with it, check extension issues")

        # Conversion des timestamps en datetime
        ecg_df['Time'] = pd.to_datetime(ecg_df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
        log_event_df['datetime'] = pd.to_datetime(log_event_df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
        
        # Conversion des timestamps en millisecondes
        ecg_df['timestamp_ms'] = ecg_df['Time'].dt.hour * 3600000 + ecg_df['Time'].dt.minute * 60000 + \
                                 ecg_df['Time'].dt.second * 1000 + ecg_df['Time'].dt.microsecond // 1000

        log_event_df['timestamp_ms'] = log_event_df['datetime'].dt.hour * 3600000 + log_event_df['datetime'].dt.minute * 60000 + \
                                       log_event_df['datetime'].dt.second * 1000 + log_event_df['datetime'].dt.microsecond // 1000

        return ecg_df, log_event_df, cog_evals_df, error_df
    else:
        print(f"Missing files for this participant : {participant_folder}")
        return None, None, None

def load_all_data(participants_root_folder):
    """
    Loads the ECG, log_event, and cognitive evaluation files for all participants in the root folder.

    Args:
        participants_root_folder (str): Path to the root directory containing all participant folders.

    Returns:
        dict: Dictionary with participant names as keys and tuples 
          (ecg_df, log_event_df, cog_evals_df) as values.
    """
    participants_data = {}
    participants_subj_data = {}
    participants_error_data = {}
    
    for participant_folder in os.listdir(participants_root_folder):
        full_path = os.path.join(participants_root_folder, participant_folder)
        if os.path.isdir(full_path):
            ecg_df, log_event_df, cog_evals_df, error_df = load_data_for_participant(full_path)
            if ecg_df is not None and log_event_df is not None:
                participants_data[participant_folder] = (ecg_df, log_event_df)
            if cog_evals_df is not None:
                participants_subj_data[participant_folder] = (cog_evals_df)
            if error_df is not None:
                participants_error_data[participant_folder] = (error_df)
            
    if os.path.exists("./Participants.csv"):
        annex_data = pd.read_csv("./Participants.csv", sep=';', encoding='iso-8859-1')
    else:
        annex_data = None
    return participants_data, participants_subj_data, participants_error_data, annex_data



def extract_errors(error_data, export=False):
    """
    Extracts and counts error occurrences from experimental log data for each participant.

    Args:
        error_data (dict): Dictionary containing error log DataFrames, with participant IDs as keys.
        export (bool, optional): If True, exports the summarized error counts to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame summarizing the number of errors by participant, segment, and error type.
    """
    data_list = []

    for participant_id, df in error_data.items():
        task_names = df.iloc[0].tolist()

        correspondance = {"(controle)": "C", "(0back)": "0B", "(2back)": "2B", 
                          "(Contrôle)": "C", "(0B)": "0B", "(2B)":"2B",
                          "(CONTRÔLE)": "C", "(0BACK)": "0B", "(2BACK)": "2B"}

        task_positions = {"C": None, "0B": None, "2B": None}
        pos_col = 0
        for task_name in task_names:
            if isinstance(task_name, str):
                if "(" in task_name:
                    splits = task_name.split("(", 1)
                    tmp = "(" + splits[1]
                    task_positions[correspondance[tmp]] = pos_col
            pos_col += 1

        error_types = df.iloc[1:, 0].unique()
        for error_type in error_types:
            for segment in task_positions.keys():
                data_list.append([participant_id, error_type, segment, 0])

        for index, row in df.iterrows():
            if index < 1:
                continue

            error_type = row.iloc[0]

            for segment, start_pos in task_positions.items():
                if start_pos is None:
                    continue
                if start_pos == 1:
                    end_pos = 8
                elif start_pos == 8:
                    end_pos = 15
                elif start_pos == 15:
                    end_pos = len(row)
                for i in range(start_pos, end_pos):
                    if row.iloc[i] == 'X':
                        data_list.append([participant_id, error_type, segment, 1])

    error_df = pd.DataFrame(data_list, columns=['Participant', 'Type', 'Segment', 'Count'])

    error_count_df = error_df.groupby(['Participant', 'Segment', 'Type']).sum().unstack(fill_value=0)
    error_count_df.columns = error_count_df.columns.droplevel()
    error_count_df['Total Errors'] = error_count_df.sum(axis=1)
    error_count_df = error_count_df.fillna(0).astype(int)

    if export: 
        error_count_df.to_csv("error_summary.csv", sep=";")

    return error_count_df


def extract_subjective_evaluations(subj_data, export=False):
    """
    Extracts and summarizes subjective workload evaluations (NASA-TLX) for each participant and segment.

    Args:
        subj_data (dict): Dictionary of DataFrames containing subjective evaluation data for each participant.
        export (bool, optional): If True, exports the processed NASA-TLX metrics to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A pivoted DataFrame containing subjective workload metrics (NASA-TLX) 
                      by participant and experimental segment.
    """
    data_list = []

    for participant, df in subj_data.items():
        for _, row in df.iterrows():
            item = row['items']
            value = row['values']
            measure, segment = item.rsplit('_', 1)
            data_list.append([participant, segment, measure, value])

    df = pd.DataFrame(data_list, columns=['Participant', 'Segment', 'NASA_TLX_Metric', 'Value'])

    df_pivot = df.pivot_table(index=['Participant', 'Segment'], columns='NASA_TLX_Metric', values='Value')

    nasa_tlx_metrics = ['mental_demand', 'physical_demand', 'temporal_demand', 'own_performance', 'effort', 'frustration_level']
    df_pivot['QtotalMW'] = df_pivot[nasa_tlx_metrics].mean(axis=1)

    segments_of_interest = ['C', '0B', '2B', 'C.1', 'C.2', '0B.1', '0B.2', '2B.1', '2B.2']
    participants = df['Participant'].unique()
    index = pd.MultiIndex.from_product([participants, segments_of_interest], names=['Participant', 'Segment'])
    df_subj = df_pivot.reindex(index)

    df_subj.to_csv("nasa_tlx_metrics.csv", sep=";")

    return df_subj






def evaluate_ecg_quality(all_segments, cleaned_segments, rpeaks, export=False, verbose=False):
    """
    Evaluates ECG signal quality before and after preprocessing for all participants and experimental segments.

    Args:
        all_segments (dict): Dictionary of raw ECG segments per participant and condition.
        cleaned_segments (dict): Dictionary of cleaned ECG segments per participant and preprocessing method.
        rpeaks (dict): Dictionary of R-peak detection results (not directly used here, kept for compatibility).
        export (bool, optional): If True, exports segment-level and global quality indices to CSV files. Defaults to False.
        verbose (bool, optional): If True, prints detailed processing information during analysis. Defaults to False.

    Returns:
        tuple: 
            - pd.DataFrame: Detailed quality ratings per participant, method, and segment.
            - pd.DataFrame: Global quality index (average quality) for each participant and method.
    """
    data = []
    segments_of_interest = ['fixation_cross', 'C', '0B', '2B']

    quality_scores = {
        "Unacceptable": 0.1,
        "Barely acceptable": 0.5,
        "Excellent": 1
    }

    participant_method_quality_indices = []

    for participant in all_segments.keys():
        total_score = 0
        count = 0
        for segment in all_segments[participant].keys():
            if segment in segments_of_interest:
                ecg_signal = all_segments[participant][segment]['EcgWaveform'].values
                quality = nk.ecg_quality(ecg_signal, method="zhao2018")
                score = quality_scores[quality]
                total_score += score
                count += 1
                if verbose:
                    print(f"Analyzing quality of segment {segment} from participant {participant}")
                data.append({
                    'Participant': participant,
                    'Method': 'no cleaning',
                    'Segment': segment,
                    'Quality': quality,
                })
        if count > 0:
            average_quality_index = round(total_score / count, 2)
            participant_method_quality_indices.append({
                'Participant': participant,
                'Method': 'no cleaning',
                'Quality Index': average_quality_index
            })

    for participant in cleaned_segments.keys():
        for method in cleaned_segments[participant].keys():
            total_score = 0
            count = 0
            for segment in cleaned_segments[participant][method].keys():
                if segment in segments_of_interest:
                    ecg_signal = cleaned_segments[participant][method][segment]['EcgWaveform'].values
                    quality = nk.ecg_quality(ecg_signal, method="zhao2018")
                    score = quality_scores[quality]
                    total_score += score
                    count += 1
                    if verbose:
                        print(f"Analyzing quality of segment {segment} cleaned by {method} from participant {participant}")
                    data.append({
                        'Participant': participant,
                        'Method': method,
                        'Segment': segment,
                        'Quality': quality,
                    })
            if count > 0:
                average_quality_index = round(total_score / count, 2)
                participant_method_quality_indices.append({
                    'Participant': participant,
                    'Method': method,
                    'Quality Index': average_quality_index
                })


    df = pd.DataFrame(data)
    df.sort_values(by=['Participant', 'Method', 'Segment'], inplace=True)
    df['Participant_Segment'] = df['Participant'] + ' : ' + df['Segment']
    q_df = df.pivot(index='Participant_Segment', columns='Method', values='Quality')
    methods_order = ['no cleaning'] + [col for col in q_df.columns if col != 'no cleaning']
    q_df = q_df[methods_order]

    global_quality_index_df = pd.DataFrame(participant_method_quality_indices, columns=['Participant', 'Method', 'Quality Index'])
    global_quality_index_df.sort_values(by=['Participant', 'Method'], inplace=True)
    gbi_df = global_quality_index_df.pivot(index='Method', columns='Participant', values='Quality Index')
    methods_order = ['no cleaning'] + [index for index in gbi_df.index if index != 'no cleaning']
    gbi_df = gbi_df.reindex(methods_order)
    
    
    if export:
        gbi_df_transposed = gbi_df.transpose()
        q_df.to_csv("Segments_Quality.csv", index=True, sep=';')
        gbi_df_transposed.to_csv("Global_Quality_Index.csv", index=True, sep=';')
    
    return df, gbi_df