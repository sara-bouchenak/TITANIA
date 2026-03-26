import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning.outliers.local_methods import LocalOLDataCleaning


class GlobalOLDataCleaning(LocalOLDataCleaning):

    def __init__(
        self,
        sensitive_attributes: list[str],
        detection_mode: str,
        correction_mode: str,
        global_stat_dict,
    ):
        super().__init__(sensitive_attributes, detection_mode, correction_mode)

        self.global_mean_dict = global_stat_dict["mean"] if "mean" in global_stat_dict.keys() else {}
        self.global_mode_dict = global_stat_dict["mode"] if "mode" in global_stat_dict.keys() else {}
        self.global_std_dict = global_stat_dict["std"] if "std" in global_stat_dict.keys() else {}
        self.global_q1_dict = global_stat_dict["q1"] if "q1" in global_stat_dict.keys() else {}
        self.global_q3_dict = global_stat_dict["q3"] if "q3" in global_stat_dict.keys() else {}

    def detect_outliers(self, X_num: pd.DataFrame) -> pd.DataFrame:
        mode = self.outliers_detection_mode
        assert mode in ["std", "iqr"]
        outliers_masks = []
        for col in X_num.columns:
            if mode == "std":
                lower_bound = self.global_mean_dict[col] - 3 * self.global_std_dict[col]
                upper_bound = self.global_mean_dict[col] + 3 * self.global_std_dict[col]
            else:
                iqr = self.global_q3_dict[col] - self.global_q1_dict[col]
                lower_bound = self.global_q1_dict[col] - 1.5 * iqr
                upper_bound = self.global_q3_dict[col] + 1.5 * iqr
            outliers_masks.append(~ X_num[col].between(lower_bound, upper_bound))
        outliers_masks = pd.concat(outliers_masks, axis=1)
        return outliers_masks

    def correct_outliers(self, data: tuple[pd.DataFrame, pd.DataFrame], numeric_cols, outliers_masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y = data
        mode = self.outliers_correction_mode
        assert mode in ["mean", "mode", "remove"]
        if mode == "remove":
            X = X[~outliers_masks.any(axis=1)]
            y = y[~outliers_masks.any(axis=1)]
        else:
            for col in numeric_cols:
                if mode == "mean":
                    val = self.global_mean_dict[col]
                    if pd.api.types.is_integer_dtype(X[col]):
                        val = int(val)
                else: # elif mode == "mode":
                    val = self.global_mode_dict[col]
                X.loc[:, col] = X[col].mask(outliers_masks[col], val)
        return X, y
