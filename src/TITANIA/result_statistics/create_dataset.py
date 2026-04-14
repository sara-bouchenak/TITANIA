import numpy as np
import os
import json
from omegaconf import OmegaConf
import pandas as pd

def compute_metrics_name_dict(columns, sensitive_attributes):

    other_columns = {}
    other_columns["info"] = []
    if "round" in columns:
        other_columns["info"].append("round")
    if "id_client" in columns:
        other_columns["info"].append("id_client")

    bias_metrics_dict = {sens_attr: [column for column in columns if sens_attr in column] for sens_attr in sensitive_attributes}
    bias_metrics_list = [bias_metric for sublist in bias_metrics_dict.values() for bias_metric in sublist]
    all_utility_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss', "training_loss"]
    utility_metrics_list = [column for column in columns if column in all_utility_metrics]

    other_columns["method_pars"] = []
    for column in columns:
        if column not in utility_metrics_list + bias_metrics_list + other_columns["info"]:
            other_columns["method_pars"].append(column)

    metrics_by_cat = bias_metrics_dict
    metrics_by_cat["utility"] = utility_metrics_list
    delete_key = [key for key, val in metrics_by_cat.items() if val == []]
    for key in delete_key:
        del metrics_by_cat[key]

    return metrics_by_cat, other_columns

def preprocess_data(df, metrics_by_cat, other_columns):
    
    for column in other_columns["method_pars"]:
        if column == "data_cleaning":
            df[column] = rename_data_cleaning_methods(df[column])
        if column == "method":
            df[column] = df[column].apply(lambda x: x if x is not np.nan else "")
            df[column] = df[column].apply(lambda x: "FedAvg" if x == "default" else x)
        df[column] = df[column].astype(str)

    if other_columns["method_pars"] != []:
        cols_without_seed = [col for col in other_columns["method_pars"] if col != "exp_seed" and col != "data_seed" ]
        df["method_name"] = df[cols_without_seed].agg(' '.join, axis=1)
        df["method_name"] = df["method_name"].apply(lambda x: x.strip())
        cols_with_seed = other_columns["method_pars"]
        df["method_name_with_seed"] = df[cols_with_seed].agg(' '.join, axis=1)
        df["method_name_with_seed"] = df["method_name_with_seed"].apply(lambda x: x.strip())

    for column in other_columns["info"]:
        df[column] = df[column].astype('int32')

    for cat, metrics in metrics_by_cat.items():
        for metric in metrics:
            if "disparate_impact" not in metric and "loss" not in metric:
                df[metric] = df[metric]*100
            if cat != "utility":
                df[metric] = df[metric].abs()

    return df

def rename_data_cleaning_methods(df_column):

    data_cleaning_mapping = {
        np.nan: "No cleaning",
        "default": "No cleaning",
        "local_LE_cleanlab_split_flip" : "LE-cleanlab-flip-split-L",
        "local_LE_cleanlab_std_flip" : "LE-cleanlab-flip-std-L",

        "local_LE_cleanlab_flip" : "LE-cleanlab-flip-L",
        "local_LE_cleanlab_nan" : "LE-cleanlab-nan-L",
        "all_errors":"all-errors",
        "multi_error_types":"all-errors",
        "multi_error_types_FL":"all-errors-FL",
        "multi_error_types_flip":"all-errors-flip",
        "multi_error_types_FL_flip":"all-errors-FL-flip",
        "local_MV_mode": "MV-mode-L",
        "local_MV_mean": "MV-mean-L",
        "local_OL_std_mean": "OL-std-mean-L",
        "local_OL_std_mode": "OL-std-mode-L",
        "local_OL_std_nan": "OL-std-nan-L",
        "local_OL_iqr_mean": "OL-iqr-mean-L",
        "local_OL_iqr_mode": "OL-iqr-mode-L",
        "local_OL_iqr_nan": "OL-iqr-nan-L",
        "global_MV_mode": "MV-mode-G",
        "global_MV_mean": "MV-mean-G",
        "global_OL_std_mean": "OL-std-mean-G",
        "global_OL_std_mode": "OL-std-mode-G",
        "global_OL_std_nan": "OL-std-nan-G",
        "global_OL_iqr_mode": "OL-iqr-mean-G",
        "global_OL_iqr_mean": "OL-iqr-mode-G",
        "global_OL_iqr_nan": "OL-iqr-nan-G",
        "FedCorr":"LE-FedCorr",
        "local_label_errors_cleanlab" : "LE-cleanlab-L",
        "local_nan_mode": "MV-mode-L",
        "local_nan_mean": "MV-mean-L",
        "local_outliers_std_mean": "OL-std-mean-L",
        "local_outliers_std_mode": "OL-std-mode-L",
        "local_outliers_std_nan": "OL-std-nan-L",
        "local_outliers_iqr_mean": "OL-iqr-mean-L",
        "local_outliers_iqr_mode": "OL-iqr-mode-L",
        "local_outliers_iqr_nan": "OL-iqr-nan-L",
        "global_nan_mode": "MV-mode-G",
        "global_nan_mean": "MV-mean-G",
        "global_outliers_std_mean": "OL-std-mean-G",
        "global_outliers_std_mode": "OL-std-mode-G",
        "global_outliers_std_nan": "OL-std-nan-G",
        "global_outliers_iqr_mode": "OL-iqr-mode-G",
        "global_outliers_iqr_mean": "OL-iqr-mean-G",
        "global_outliers_iqr_nan": "OL-iqr-nan-G",
        "cafe": "MV-cafe",

        "local_stat_nan_mode": "MV-mode-L",
        "local_stat_nan_mean": "MV-mean-L",
        "local_stat_outliers_std_mean": "OL-std-mean-L",
        "local_stat_outliers_std_mode": "OL-std-mode-L",
        "local_stat_outliers_std_nan": "OL-std-nan-L",
        "local_stat_outliers_iqr_mean": "OL-iqr-mean-L",
        "local_stat_outliers_iqr_mode": "OL-iqr-mode-L",
        "local_stat_outliers_iqr_nan": "OL-iqr-nan-L",
        "global_stat_nan_mode": "MV-mode-G",
        "global_stat_nan_mean": "MV-mean-G",
        "global_stat_outliers_std_mean": "OL-std-mean-G",
        "global_stat_outliers_std_mode": "OL-std-mode-G",
        "global_stat_outliers_iqr_mode": "OL-iqr-mode-G",
        "global_stat_outliers_iqr_mean": "OL-iqr-mean-G",
        "global_stat_outliers_iqr_nan": "OL-iqr-nan-G",
    }

    return df_column.apply(lambda x: data_cleaning_mapping[x] if x in data_cleaning_mapping.keys() else x)

def load_df_multirun(exp_dir, metrics_type):
    json_metrics_paths = []
    for root, dirs, files in os.walk(exp_dir):
        #print(files)
        for file in files:
            if file.endswith("results.json"):
                json_metrics_paths.append(os.path.join(root, file))

    df = []
    for path in json_metrics_paths:
        df_run = load_df(path, metrics_type)
        pars = path.split(exp_dir)[1].replace("/", ",").replace("\\", ",").split(",")[1:-1]
        for i in range(30):
            pars[0]=pars[0].replace(f"{29-i}_","")
        pars = {par.split("=")[0]: par.split("=")[1] for par in pars}
        for key, val in pars.items():
                df_run[key] = val
        df.append(df_run)
    #print("df",df)
    df = pd.concat(df, axis=0, ignore_index=True)
    return df

def load_cfg_multirun(exp_dir):
    config_paths = []
    for root, dirs, files in os.walk(exp_dir):
        if ".hydra" not in root:
            for file in files:
                if file.endswith("config.yaml"):
                    config_paths.append(os.path.join(root, file))

    cfg = OmegaConf.load(config_paths[0])
    return cfg

def load_df(json_metrics_path, metrics_type):
    with open(json_metrics_path, "r") as f:
            metrics_dict = json.load(f)

    metrics_subdict = metrics_dict[metrics_type]
    if metrics_type == "perf_global":
        df = compute_perf_global_metrics(metrics_subdict)
    elif metrics_type == "perf_locals":
        df = compute_perf_locals_metrics(metrics_subdict)
    elif metrics_type == "perf_prefit" or metrics_type == "perf_postfit":
        df = compute_perf_prefit_postfit_metrics(metrics_subdict)
    elif metrics_type == "custom_fields":
        df = compute_custom_fields_metrics(metrics_subdict)
    else:
        raise TypeError("metrics_type '{}' is unknown!".format(metrics_type))
    return df

def compute_perf_global_metrics(perf_global_dict: dict):
    lines = []
    columns = []
    for round, metrics in perf_global_dict.items():
        if columns == []:
            columns = list(metrics.keys())
        line = [metric for metric in metrics.values()]
        line.append(round)
        lines.append(line)
    columns.append("round")
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    return df_metrics

def compute_perf_locals_metrics(perf_local_dict: dict):
    lines = []
    columns = []
    for round, clients_dict in perf_local_dict.items():
        for id_client, metrics in clients_dict['null'].items():
            if columns == []:
                columns = list(metrics.keys())
            line = [metric for metric in metrics.values()]
            line.append(id_client)
            line.append(round)
            lines.append(line)
    columns.extend(["id_client", "round"])
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    return df_metrics

def compute_perf_prefit_postfit_metrics(perf_dict):
    lines = []
    columns = []
    for round, clients_dict in perf_dict.items():
        for id_client, metrics in clients_dict.items():
            if columns == []:
                columns = list(metrics.keys())
            line = [metric for metric in metrics.values()]
            line.append(id_client)
            line.append(round)
            lines.append(line)
    columns.extend(["id_client", "round"])
    df_metrics = pd.DataFrame(data=lines, columns=columns)
    return df_metrics

def compute_custom_fields_metrics(custom_fields_dict):

    glob_metrics = custom_fields_dict["-1"]
    lines_glob = list(glob_metrics.values())
    columns_glob = list(glob_metrics.keys())
    df = pd.DataFrame(data=[lines_glob], columns=columns_glob)

    lines_others = []
    columns_others = []
    for round, subdict in custom_fields_dict.items():
        if round != "-1":
            if columns_others == []:
                columns_others = list(subdict.keys())
            line = list(subdict.values())
            line.append(round)
            lines_others.append(line)

    if lines_others != []:
        columns_others.append("round")
        df_rounds = pd.DataFrame(data=lines_others, columns=columns_others)
        df = pd.concat([df, df_rounds], axis=0)

    return df
def main(dataset_names,exp_names):
    dataset_names= [item for item in dataset_names.split(',')]
    exp_names= [item for item in exp_names.split(',')]
    for dataset_name in dataset_names:
        for exp_name in exp_names:
            metrics_type = "perf_global"
            #dataset_name="ARS" 
            #exp_name="FL_non_iid_settings"
            main_dir="traces"
            exp_dir=main_dir+"/"+exp_name+"/dataset="+dataset_name
            print(exp_dir)
            df = load_df_multirun(exp_dir, metrics_type)

            cfg = load_cfg_multirun(exp_dir)
            sensitive_attributes = cfg.data.dataset.sensitive_attributes

            metrics_by_cat, other_columns = compute_metrics_name_dict(df.columns.tolist(), sensitive_attributes)
            df = preprocess_data(df, metrics_by_cat, other_columns)
            df.to_csv(main_dir+"/"+exp_name+"/"+dataset_name+".csv", index=False) 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--dataset', required=True, help='comma seperated list: name of the dataset being used')
    parser.add_argument('--experiment',required=True, help='comma seperated list: name of experiment being tested')
    args = parser.parse_args()
    main(args.dataset,args.experiment)
