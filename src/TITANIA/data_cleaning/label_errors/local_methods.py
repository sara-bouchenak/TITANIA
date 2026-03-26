import pandas as pd
import numpy as np

import cleanlab
from sklearn.linear_model import SGDClassifier

from src.FL_core.data_loading.data_processing import one_hot_encoding, normalization, convert_bool_and_cat_to_num
from src.TITANIA.data_cleaning.label_errors.default import LabelErrorsDataCleaningMethod


class LocalLEDataCleaning(LabelErrorsDataCleaningMethod):

    def __init__(
        self,
        sensitive_attributes: list[str],
        detection_mode: str,
        correction_mode: str,
        seed: int | None = None,
    ):
        super().__init__(sensitive_attributes)
        self.label_errors_detection_mode = detection_mode
        self.label_errors_correction_mode = correction_mode
        self.seed = seed

    def detect_label_errors(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        mode = self.label_errors_detection_mode
        if mode == "cleanlab_standard":
            label_errors_masks = detect_mislabeled_via_cleanlab_standard(data, seed=self.seed)
        elif mode == "cleanlab_split":
            label_errors_masks = detect_mislabeled_via_cleanlab_split(data, seed=self.seed)
        else:
            raise ValueError("Not implemented "+mode)
        return label_errors_masks

    def correct_label_errors(self, data: tuple[pd.DataFrame, pd.DataFrame], label_errors_masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        mode = self.label_errors_correction_mode
        assert mode in ["binary_flip", "remove"]

        X, y = data
        if mode == "binary_flip":
            X_replaced = X
            y_replaced = y.mask(label_errors_masks, ~y)
        elif mode=="remove":
            X_replaced=X[~label_errors_masks.any(axis=1)]
            y_replaced=y[~label_errors_masks.any(axis=1)]
        else:
            X_replaced = X
            y_replaced = y
        return X_replaced, y_replaced


def detect_mislabeled_via_cleanlab_standard(data: tuple[pd.DataFrame, pd.DataFrame], seed) -> pd.DataFrame:
    ohe_data = one_hot_encoding({"all": data})
    numeric_data = convert_bool_and_cat_to_num(ohe_data)
    normalized_data = normalization({"clients_train": numeric_data}, [])
    X, y = normalized_data["clients_train"]["all"]
    if y.iloc[:, 0].nunique() > 1 :
        model = SGDClassifier(loss='log_loss', random_state=seed).fit(X, y.iloc[:, 0])
        cl = cleanlab.classification.CleanLearning(model, seed=seed)
        label_issues = cl.find_label_issues(X, y.iloc[:, 0])
        label_errors_masks = label_issues[["is_label_issue"]]
        label_errors_masks = label_errors_masks.rename(columns={"is_label_issue": y.columns[0]})
    else:
        label_errors_masks = y.copy()
        label_errors_masks.iloc[:, 0] = False
    return label_errors_masks

def detect_mislabeled_via_cleanlab_split(data: tuple[pd.DataFrame, pd.DataFrame], seed) -> pd.DataFrame:
    ohe_data = one_hot_encoding({"all": data})
    numeric_data = convert_bool_and_cat_to_num(ohe_data)
    normalized_data = normalization({"clients_train": numeric_data}, [])
    X, y = normalized_data["clients_train"]["all"]
    X1=X[:(len(X)//2)]
    X2=X[(len(X)//2):]
    y1=y[:(len(X)//2)]
    y2=y[(len(X)//2):]
    if y1.iloc[:, 0].nunique() > 1 :
        model = SGDClassifier(loss='log_loss', random_state=seed).fit(X1, y1.iloc[:, 0])
        cl = cleanlab.classification.CleanLearning(model, seed=seed)
        label_issues = cl.find_label_issues(X2, y2.iloc[:, 0])
        label_errors_masks2= label_issues[["is_label_issue"]]
        label_errors_masks2 = label_errors_masks2.rename(columns={"is_label_issue": y2.columns[0]})
    else:
        label_errors_masks2 = y2.copy()
        label_errors_masks2.iloc[:, 0] = False
    if y2.iloc[:, 0].nunique() > 1 :
        model = SGDClassifier(loss='log_loss', random_state=seed).fit(X2, y2.iloc[:, 0])
        cl = cleanlab.classification.CleanLearning(model, seed=seed)
        label_issues = cl.find_label_issues(X1, y1.iloc[:, 0])
        label_errors_masks1= label_issues[["is_label_issue"]]
        label_errors_masks1 = label_errors_masks1.rename(columns={"is_label_issue": y1.columns[0]})
    else:
        label_errors_masks1 = y1.copy()
        label_errors_masks1.iloc[:, 0] = False
    label_errors_masks=pd.concat([label_errors_masks1,label_errors_masks2], ignore_index=True, sort=False)
    return label_errors_masks
