import time
import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning.label_errors import LabelErrorsDataCleaningMethods
from src.TITANIA.data_cleaning.missing_values import MissingValuesDataCleaningMethods
from src.TITANIA.data_cleaning.outliers import OutliersDataCleaningMethods


def clean_data(data, cfg, sensitive_attributes):

    data_cleaning_metrics = {}

    start_cleaning_time = time.time()

    if ("outliers" in cfg.keys() and cfg.outliers.name == "global") or ("missing_values" in cfg.keys() and cfg.missing_values.name == "global"):
        global_stat_dict = compute_global_stat_values(data["clients_train"], cfg)
    else:
        global_stat_dict = {}

    if ("order" in cfg.keys()) and (cfg.order == "flip_order"):

        if "missing_values" in cfg.keys():
            print("started missing value error cleaning")
            mv_method = MissingValuesDataCleaningMethods.get(**cfg.missing_values, sensitive_attributes=sensitive_attributes, global_stat_dict=global_stat_dict)
            data, mv_metrics = mv_method.clean_errors(data)
            data_cleaning_metrics["missing_values"] = mv_metrics

        if "outliers" in cfg.keys():
            print("started outlier error cleaning")
            ol_method = OutliersDataCleaningMethods.get(**cfg.outliers, sensitive_attributes=sensitive_attributes, global_stat_dict=global_stat_dict)
            data, ol_metrics = ol_method.clean_errors(data)
            data_cleaning_metrics["outliers"] = ol_metrics

        if "label_errors" in cfg.keys():
            print("started label error cleaning")
            le_method = LabelErrorsDataCleaningMethods.get(**cfg.label_errors, sensitive_attributes=sensitive_attributes)
            data, le_metrics = le_method.clean_errors(data)
            data_cleaning_metrics["label_errors"] = le_metrics

    else:

        if "label_errors" in cfg.keys():
            print("started label error cleaning")
            le_method = LabelErrorsDataCleaningMethods.get(**cfg.label_errors, sensitive_attributes=sensitive_attributes)
            data, le_metrics = le_method.clean_errors(data)
            data_cleaning_metrics["label_errors"] = le_metrics

        if "outliers" in cfg.keys():
            print("started outlier error cleaning")
            ol_method = OutliersDataCleaningMethods.get(**cfg.outliers, sensitive_attributes=sensitive_attributes, global_stat_dict=global_stat_dict)
            data, ol_metrics = ol_method.clean_errors(data)
            data_cleaning_metrics["outliers"] = ol_metrics

        if "missing_values" in cfg.keys():
            print("started missing value error cleaning")
            mv_method = MissingValuesDataCleaningMethods.get(**cfg.missing_values, sensitive_attributes=sensitive_attributes, global_stat_dict=global_stat_dict)
            data, mv_metrics = mv_method.clean_errors(data)
            data_cleaning_metrics["missing_values"] = mv_metrics

    end_cleaning_time = time.time()
    cleaning_time = end_cleaning_time - start_cleaning_time
    data_cleaning_metrics["total_cleaning_time"] = cleaning_time

    return data, data_cleaning_metrics

def compute_global_stat_values(clients_data_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]], cfg):

    if "missing_values" in cfg.keys() and cfg.missing_values.name == "global":
        missing_values_correction_mode = cfg.missing_values.correction_mode
    else:
        missing_values_correction_mode = None

    if "outliers" in cfg.keys() and cfg.outliers.name == "global":
        outliers_detection_mode = cfg.outliers.detection_mode
        outliers_correction_mode = cfg.outliers.correction_mode
    else:
        outliers_detection_mode = None
        outliers_correction_mode = None

    global_stat_dict = {"mean": {}, "mode": {}, "std": {}, "q1": {}, "q3": {}}

    X_global = []
    y_global = []
    for id_client, client_tr_data in clients_data_dict.items():
        X, y = client_tr_data
        X_global.append(X)
        y_global.append(y)
    X_global = pd.concat(X_global, axis=0, ignore_index=True)
    y_global = pd.concat(y_global, axis=0, ignore_index=True)

    numeric_cols = X_global.select_dtypes(include=np.number).columns.tolist()
    for col in X_global.columns:
        if col in numeric_cols:
            if outliers_detection_mode == "std":
                global_stat_dict["std"][col] = X_global[col].std()
            elif outliers_detection_mode == "iqr":
                global_stat_dict["q1"][col] = X_global[col].quantile(q=0.25)
                global_stat_dict["q3"][col] = X_global[col].quantile(q=0.75)
            if outliers_detection_mode is not None or missing_values_correction_mode == "mean":
                global_stat_dict["mean"][col] = X_global[col].mean()
            if outliers_correction_mode == "mode" or missing_values_correction_mode == "mode":
                global_stat_dict["mode"][col] = X_global[col].mode()[0]
        else:
            if missing_values_correction_mode is not None:
                global_stat_dict["mode"][col] = X_global[col].mode()[0]

    if missing_values_correction_mode is not None:
        numeric_cols = y_global.select_dtypes(include=np.number).columns.tolist()
        for col in y_global.columns:
            if missing_values_correction_mode == "mean" and col in numeric_cols:
                global_stat_dict["mean"][col] = y_global[col].mean()
            else:
                global_stat_dict["mode"][col] = y_global[col].mode()[0]

    return global_stat_dict
