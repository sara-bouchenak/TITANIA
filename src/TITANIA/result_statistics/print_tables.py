import os
import argparse
from typing import Any
import pandas as pd

from load_metrics import load_df_multirun
from utils import compute_metrics_name_dict, preprocess_data
from t_tests import compute_t_tests, aggregate_df_t_tests_by
from t_tests import aggregate_df_t_tests_for_joint_impact

def main(exp_name) -> Any:
    project_dir = "."
    exp_dir = os.path.join(project_dir, "traces", exp_name)

    df, metrics_by_cat, other_columns = load_df_results(exp_dir)

    #df_agg = aggregate_metrics(df, metrics_by_cat, other_columns, n_last_rounds_agg=20)
    df_agg = keep_metrics_n_last_rounds(df, metrics_by_cat, other_columns, n_last_rounds=20)

    df_t_tests = compute_t_tests(
        df_agg,
        metrics_by_cat,
        other_columns,
        t_test_column="data_cleaning",
        baseline_name="default",
        significant_th=0.01,
    )

    df_joint = aggregate_df_t_tests_for_joint_impact(df_t_tests, metrics_by_cat, other_columns)
    print(df_joint)
    print()

    df_metrics_pivot = aggregate_df_t_tests_by(df_t_tests, by="metric")
    print(df_metrics_pivot)
    print()

    df_metrics_pivot = aggregate_df_t_tests_by(df_t_tests, by="metric_wo_sa")
    print(df_metrics_pivot)
    print()

    df_cleaning_pivot = aggregate_df_t_tests_by(df_t_tests, by="data_cleaning")
    print(df_cleaning_pivot)
    print()

    df_dataset_pivot = aggregate_df_t_tests_by(df_t_tests, by="dataset")
    print(df_dataset_pivot)
    print()

    df_model_pivot = aggregate_df_t_tests_by(df_t_tests, by="model")
    print(df_model_pivot)
    print()

def load_df_results(exp_dir: str):
    df = load_df_multirun(exp_dir, "perf_global")
    metrics_by_cat, other_columns = compute_metrics_name_dict(df.columns.tolist())
    df = preprocess_data(df, metrics_by_cat, other_columns)
    return df, metrics_by_cat, other_columns

def aggregate_metrics(
    df: pd.DataFrame,
    metrics_by_cat: dict,
    other_columns: dict,
    n_last_rounds_agg: int,
):
    df = df[df["source"] == "server"]
    metrics = [metric for key, sublist in metrics_by_cat.items() for metric in sublist]
    lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds_agg][metrics].mean()
    agg_columns = other_columns["method_pars"] + ["method_name", "method_name_with_seed"]
    df_agg = df.groupby(agg_columns, as_index=False).apply(lambda_agg)
    return df_agg

def keep_metrics_n_last_rounds(
    df: pd.DataFrame,
    metrics_by_cat: dict,
    other_columns: dict,
    n_last_rounds: int,
):
    df = df[df["source"] == "server"]
    lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds]
    agg_columns = other_columns["method_pars"] + ["method_name", "method_name_with_seed"]
    df_agg = df.groupby(agg_columns, as_index=False).apply(lambda_agg)
    return df_agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    args = parser.parse_args()
    main(args.exp_name)
