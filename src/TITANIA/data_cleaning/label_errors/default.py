import pandas as pd

from src.TITANIA.data_cleaning import DataCleaningMethod


class LabelErrorsDataCleaningMethod(DataCleaningMethod):

    def __init__(self, sensitive_attributes: list[str]):
        super().__init__(sensitive_attributes, "label_errors")

    def clean_errors_dataloader(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data == None:
            return None
        else:
            X_new, y_new = self.handle_label_errors(data)
            data_cleaned = (X_new, y_new)
            return data_cleaned

    def handle_label_errors(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, _ = data
        self.count_n_samples_by_sensitive_attributes(X)
        label_errors_masks = self.detect_label_errors(data)
        self.count_n_detected_errors_by_sensitive_attributes(label_errors_masks, X[self.sensitive_attributes])
        X_replaced, y_replaced = self.correct_label_errors(data, label_errors_masks)
        return X_replaced, y_replaced

    def detect_label_errors(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        _, y = data
        label_errors_masks = y.copy()
        label_errors_masks.iloc[:, :] = False
        return label_errors_masks

    def correct_label_errors(self, data: tuple[pd.DataFrame, pd.DataFrame], label_errors_masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return data

    def count_n_detected_errors_by_sensitive_attributes(self, label_mask: pd.DataFrame, X_sa: pd.DataFrame):
        for sens_attr in self.sensitive_attributes:
            n_label_errors_sa = label_mask.any(axis=1).groupby(X_sa[sens_attr]).sum()
            n_group_A = n_label_errors_sa.loc[True] if True in n_label_errors_sa.index else 0
            self.n_detected_errors[sens_attr]["group_A"] += n_group_A
            n_group_B = n_label_errors_sa.loc[False] if False in n_label_errors_sa.index else 0
            self.n_detected_errors[sens_attr]["group_B"] += n_group_B
