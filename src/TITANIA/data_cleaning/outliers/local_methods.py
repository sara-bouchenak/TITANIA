import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning.outliers.default import OutliersDataCleaningMethod


class LocalOLDataCleaning(OutliersDataCleaningMethod):

    def __init__(
        self,
        sensitive_attributes: list[str],
        detection_mode: str,
        correction_mode: str,
    ):
        super().__init__(sensitive_attributes)
        self.outliers_detection_mode = detection_mode
        self.outliers_correction_mode = correction_mode

    def detect_outliers(self, X_num: pd.DataFrame) -> pd.DataFrame:
        mode = self.outliers_detection_mode
        assert mode in ["std", "iqr"]
        outliers_masks = []
        for col in X_num.columns:
            if mode == "std":
                lower_bound = X_num[col].mean() - 3 * X_num[col].std()
                upper_bound = X_num[col].mean() + 3 * X_num[col].std()
            else:
                iqr = X_num[col].quantile(q=0.75) - X_num[col].quantile(q=0.25)
                lower_bound = X_num[col].quantile(q=0.25) - 1.5 * iqr
                upper_bound = X_num[col].quantile(q=0.75) + 1.5 * iqr
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
                    val = X[col].mean()
                    if pd.api.types.is_integer_dtype(X[col]):
                        val = int(val)
                else: # elif mode == "mode":
                    val = X[col].mode()[0]
                X.loc[:, col] = X[col].mask(outliers_masks[col], val)
        return X, y
