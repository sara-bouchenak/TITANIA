import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning import DataCleaningMethod


class OutliersDataCleaningMethod(DataCleaningMethod):

    def __init__(self, sensitive_attributes: list[str]):
        super().__init__(sensitive_attributes, "outliers")

    def clean_errors_dataloader(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data == None:
            return None
        else:
            X_new, y_new = self.handle_outliers(data)
            data_cleaned = (X_new, y_new)
            return data_cleaned

    def handle_outliers(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y = data
        self.count_n_samples_by_sensitive_attributes(X)
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 0:
            outliers_masks = self.detect_outliers(X[numeric_cols])
            self.count_n_detected_errors_by_sensitive_attributes(outliers_masks, X[self.sensitive_attributes])
            X, y = self.correct_outliers(data, numeric_cols, outliers_masks)
        return X, y 

    def detect_outliers(self, X_num: pd.DataFrame) -> pd.DataFrame:
        outliers_masks = X_num.copy()
        outliers_masks.iloc[:, :] = False
        return outliers_masks

    def correct_outliers(self, data: tuple[pd.DataFrame, pd.DataFrame], numeric_cols, outliers_masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return data

    def count_n_detected_errors_by_sensitive_attributes(self, outliers_mask: pd.DataFrame, X_sa: pd.DataFrame):
        for sens_attr in self.sensitive_attributes:
            n_outliers_sa = outliers_mask.any(axis=1).groupby(X_sa[sens_attr]).sum()
            n_group_A = n_outliers_sa.loc[True] if True in n_outliers_sa.index else 0
            self.n_detected_errors[sens_attr]["group_A"] += n_group_A
            n_group_B = n_outliers_sa.loc[False] if False in n_outliers_sa.index else 0
            self.n_detected_errors[sens_attr]["group_B"] += n_group_B
