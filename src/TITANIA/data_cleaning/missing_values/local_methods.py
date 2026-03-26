import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning.missing_values.default import MissingValuesDataCleaningMethod


class LocalMVDataCleaning(MissingValuesDataCleaningMethod):

    def __init__(
        self,
        sensitive_attributes: list[str],
        correction_mode: str,
    ):
        super().__init__(sensitive_attributes)
        self.missing_values_correction_mode = correction_mode

    def handle_missing_values(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y = data
        self.count_n_samples_by_sensitive_attributes(X)
        self.count_n_detected_errors_by_sensitive_attributes(data)
        X_cleaned = self.correct_missing_values(X.copy())
        y_cleaned = self.correct_missing_values(y.copy())
        return (X_cleaned, y_cleaned)

    def correct_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = self.missing_values_correction_mode
        assert mode in ["mean", "mode"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if mode == "mean" and col in numeric_cols:
                    val = df[col].mean()
                    if pd.api.types.is_integer_dtype(df[col]):
                        val = int(val)
                else:
                    val = df[col].mode()[0]
                df.loc[:, col] = df[col].fillna(val)
        return df
