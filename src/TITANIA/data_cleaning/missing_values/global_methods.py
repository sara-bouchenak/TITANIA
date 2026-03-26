import pandas as pd
import numpy as np

from src.TITANIA.data_cleaning.missing_values.local_methods import LocalMVDataCleaning


class GlobalMVDataCleaning(LocalMVDataCleaning):

    def __init__(
        self,
        sensitive_attributes: list[str],
        correction_mode: str,
        global_stat_dict,
    ):
        super().__init__(sensitive_attributes, correction_mode)

        self.global_mean_dict = global_stat_dict["mean"] if "mean" in global_stat_dict.keys() else {}
        self.global_mode_dict = global_stat_dict["mode"] if "mode" in global_stat_dict.keys() else {}

    def correct_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = self.missing_values_correction_mode
        assert mode in ["mean", "mode"]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if mode == "mean" and col in numeric_cols:
                    val = self.global_mean_dict[col]
                    if pd.api.types.is_integer_dtype(df[col]):
                        val = int(val)
                else:
                    val = self.global_mode_dict[col]
                df.loc[:, col] = df[col].fillna(val)
        return df
