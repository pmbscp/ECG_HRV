from data_preprocessing import load_all_data
from src.segmentation import segment_ecg_by_condition
from src.hrv_metrics import compute_hrv_features


if __name__ == "__main__":
    df_raw = load_all_data("data/")
    df_segments = segment_ecg_by_condition(df_raw, "data/logs/")
    df_hrv = compute_hrv_features(df_segments)
