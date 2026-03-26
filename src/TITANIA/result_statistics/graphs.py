from typing import Any
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def main(dataset_names,exp_names):
    if not os.path.isdir("plots"):
        os.makedirs("plots")
    dataset_names= [item for item in dataset_names.split(',')]
    exp_names= [item for item in exp_names.split(',')]
    fair_metrics_g_non_iid = ["DcI$_{gender}$","SPD$_{gender}$","AOD$_{gender}$","EOD$_{gender}$","DI$_{gender}$"]
    fair_metrics_a_non_iid = ["DcI$_{age}$","SPD$_{age}$","AOD$_{age}$","EOD$_{age}$","DI$_{age}$"]
    fair_metrics_r_non_iid = ["DcI$_{race}$","SPD$_{race}$","AOD$_{race}$","EOD$_{race}$","DI$_{race}$"]
    fair_metrics_g = [["DcI$_{gender}$","SPD$_{gender}$"],["AOD$_{gender}$","EOD$_{gender}$"],["DI$_{gender}$","F1-score"]]
    fair_metrics_a = [["DcI$_{age}$","SPD$_{age}$"],["AOD$_{age}$","EOD$_{age}$"],["DI$_{gender}$","F1-score"]]
    fair_metrics_r = [["DcI$_{race}$","SPD$_{race}$"],["AOD$_{race}$","EOD$_{race}$"],["DI$_{gender}$","F1-score"]]

    for exp_name in exp_names:
        if not os.path.isdir("plots/"+exp_name):
            os.makedirs("plots/"+exp_name)
        
        for dataset_name in dataset_names:
            print(dataset_name,exp_name)
            if not os.path.isdir("plots/"+exp_name+"/"+dataset_name):
                os.makedirs("plots/"+exp_name+"/"+dataset_name)
            if dataset_name=="Adult" or dataset_name=="KDD":
                fair=fair_metrics_g+fair_metrics_a+fair_metrics_r
            elif dataset_name=="Heart":
                fair=fair_metrics_g+fair_metrics_a
            elif dataset_name=="MEPS":
                fair=fair_metrics_g+fair_metrics_r
            elif dataset_name=="ARS":
                fair=fair_metrics_g
            else:
                raise ValueError("bad datasets"+dataset_name)

            if exp_name=="FL_non_iid_settings" or exp_name=="example":
                
                if dataset_name=="Adult" or dataset_name=="KDD":
                    fair=fair_metrics_g_non_iid+fair_metrics_a_non_iid+fair_metrics_r_non_iid
                elif dataset_name=="Heart":
                    fair=fair_metrics_g_non_iid+fair_metrics_a_non_iid
                elif dataset_name=="MEPS":
                    fair=fair_metrics_g_non_iid+fair_metrics_r_non_iid
                elif dataset_name=="ARS":
                    fair=fair_metrics_g_non_iid
                else:
                    raise ValueError("bad datasets"+dataset_name)
                if not os.path.isdir("plots/"+f"{exp_name}2"):
                    os.makedirs("plots/"+f"{exp_name}2")
                if not os.path.isdir("plots/"+f"{exp_name}2"+"/"+dataset_name):
                    os.makedirs("plots/"+f"{exp_name}2"+"/"+dataset_name)
                plot_non_iid(csv_path=f"./traces/{exp_name}/{dataset_name}.csv", plot_dir=f"./plots/{exp_name}/{dataset_name}",fair_metrics=fair,include_x=False,include_all=True,include_o=True)
                plot_non_iid(csv_path=f"./traces/{exp_name}/{dataset_name}.csv", plot_dir=f"./plots/{exp_name}2/{dataset_name}",fair_metrics=fair,include_x=True,include_all=True,include_o=False)
            elif exp_name=="error_rates":
                plot_error_rates_from_csv(csv_path=f"./traces/error_rates/{dataset_name}.csv", plot_dir=f"./plots/error_rates/{dataset_name}",fairness_metrics=fair)

            elif exp_name=="bias_mitigation":
                plot_bias_mitigation(csv_path=f"./traces/bias_mitigation/{dataset_name}.csv", plot_dir=f"./plots/bias_mitigation/{dataset_name}",fairness_metrics=fair)
            else:
                raise ValueError("bad experiment"+exp_name)

    """
    plot_bias_mitigation(csv_path="./bias_mitigation/adult.csv", plot_dir="./bias_mitigation/adult_plots",fairness_metrics=(fair_metrics_g+fair_metrics_a+fair_metrics_r))
    plot_bias_mitigation(csv_path="./bias_mitigation/kdd.csv", plot_dir="./bias_mitigation/kdd_plots",fairness_metrics=(fair_metrics_g+fair_metrics_a+fair_metrics_r))

    plot_bias_mitigation(csv_path="./bias_mitigation/heart.csv", plot_dir="./bias_mitigation/heart_plots",fairness_metrics=(fair_metrics_g+fair_metrics_a))
    plot_bias_mitigation(csv_path="./bias_mitigation/meps.csv", plot_dir="./bias_mitigation/meps_plots",fairness_metrics=(fair_metrics_g+fair_metrics_r))
    plot_bias_mitigation(csv_path="./bias_mitigation/ars.csv", plot_dir="./bias_mitigation/ars_plots",fairness_metrics=(fair_metrics_g))

    plot_error_rates_from_csv(csv_path="./error_rates/heart.csv", plot_dir="./error_rates/heart_plots",fairness_metrics=(fair_metrics_g+fair_metrics_a))
    plot_error_rates_from_csv(csv_path="./error_rates/meps.csv", plot_dir="./error_rates/meps_plots",fairness_metrics=(fair_metrics_g+fair_metrics_r))
    plot_error_rates_from_csv(csv_path="./error_rates/ars.csv", plot_dir="./error_rates/ars_plots",fairness_metrics=(fair_metrics_g))
    plot_error_rates_from_csv(csv_path="./error_rates/kdd.csv", plot_dir="./error_rates/kdd_plots",fairness_metrics=(fair_metrics_g+fair_metrics_a+fair_metrics_r))

    plot_non_iid(csv_path="./FL_non_iid_settings/heart.csv", plot_dir="./FL_non_iid_settings/heart_plots",fair_metrics=(fair_metrics_g+fair_metrics_a))
    plot_non_iid(csv_path="./FL_non_iid_settings/meps.csv", plot_dir="./FL_non_iid_settings/meps_plots",fair_metrics=(fair_metrics_g+fair_metrics_r))
    plot_non_iid(csv_path="./FL_non_iid_settings/ars.csv", plot_dir="./FL_non_iid_settings/ars_plots",fair_metrics=(fair_metrics_g))
    plot_non_iid(csv_path="./FL_non_iid_settings/kdd.csv", plot_dir="./FL_non_iid_settings/kdd_plots",fair_metrics=(fair_metrics_g+fair_metrics_a+fair_metrics_r))
    plot_dir = "./FL_non_iid_settings/meps_plots"
    csv_path = "./FL_non_iid_settings/MEPS.csv"
    plot_non_iid(plot_dir,csv_path)
    """
def plot_error_rates_from_csv(csv_path, plot_dir,fairness_metrics):

    df = pd.read_csv(csv_path)
    #print(df.columns)
    df=  df.rename(columns={"precision": "Precision","recall":"Recall","f1":"F1-score","F1":"F1-score","accuracy":"Accuracy","gender_discr_index":"DcI$_{gender}$","gender_spd":"SPD$_{gender}$","gender_avg_odds": "AOD$_{gender}$","gender_eod": "EOD$_{gender}$","gender_disparate_impact": "DI$_{gender}$","age_discr_index":"DcI$_{age}$","age_spd":"SPD$_{age}$","age_avg_odds": "AOD$_{age}$","age_eod": "EOD$_{age}$","age_disparate_impact": "DI$_{age}$","race_discr_index":"DcI$_{race}$","race_spd":"SPD$_{race}$","race_avg_odds": "AOD$_{race}$","race_eod": "EOD$_{race}$","race_disparate_impact": "DI$_{race}$"})
    perf_metrics=[["Accuracy","Recall"],["F1-score","Precision"]]
    #fair_metrics=["SPD$_{gender}$"]

    metrics_list = perf_metrics + fairness_metrics
    #metrics_list = [["Precision", "Recall"],['F1',"Accuracy"],["SPD$_{gender}$","DcI$_{gender}$"],["EOD$_{gender}$","AOD$_{gender}$"],["DI$_{gender}$","F1-score"],["SPD$_{age}$","DcI$_{age}$"],["EOD$_{age}$","AOD$_{age}$"],["DI$_{age}$","F1-score"]]
    #,["SPD$_{race}$","DcI$_{race}$"],["EOD$_{race}$","AOD$_{race}$"],["DI$_{race}$","F1-score"]]#
    for metrics in metrics_list:
        
        n_last_rounds_agg = 10
        lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds_agg][metrics].mean()
        df_agg = df.groupby(["injected_label_errors_perc", "exp_seed"], as_index=False).apply(lambda_agg)
        error_mapping = {
            0.0: "0%",
            "0.0": "0%",
            0.1: "10%",
            0.2: "20%",
            0.3: "30%",
            0.4: "40%",
        }
        df_agg["error_rate"] = df_agg["injected_label_errors_perc"].apply(lambda x: error_mapping[x])
        df2 = df_agg#.rename(columns={"precision": "Precision","gender_avg_odds": "AOD$_{gender}$"})

        FIGSIZE = (5, 4)
        XTICKS_FONT_SIZE = 18
        LABELS_FONT_SIZE = 20
        VALUE_FONT_SIZE = 16
        BAR_WIDTH = 0.5
        GREEN_2  = '#7FB77E'
        #print()
        max_val=math.ceil(df2[metrics[0]].max())
        min_val=min(math.floor(df2[metrics[0]].min()),max_val-5)
        small_gap=(max_val-min_val)<30
        if small_gap:
            min_val=math.floor(min_val/5)*5
        else:
            min_val=math.floor(min_val/10)*10

        if min_val<0:
            max_val-=min_val
            min_val=0
        if metrics[0] not in ["Precision", "Recall",'F1',"Accuracy"]:
            min_val=0
        if small_gap:
            max_val=math.ceil(max_val/5)*5
        else:
            max_val=math.ceil(max_val/10)*10

        max_val2=math.ceil(df2[metrics[1]].max())
        min_val2=min(math.floor(df2[metrics[1]].min()),max_val2-5)
        small_gap=(max_val2-min_val2)<30
        if small_gap:
            min_val2=math.floor(min_val2/5)*5
        else:
            min_val2=math.floor(min_val2/10)*10

        if min_val2<0:
            max_val2-=min_val2
            min_val2=0
        if metrics[1] not in ["Precision", "Recall",'F1',"Accuracy"]:
            min_val2=0
        if small_gap:
            max_val2=math.ceil(max_val2/5)*5
        else:
            max_val2=math.ceil(max_val2/10)*10

        if max_val-min_val<6:
            y_step1=1
        elif max_val-min_val<11:
            y_step1=2
        elif max_val-min_val<31:
            y_step1=5
        elif max_val-min_val<51:
            y_step1=10
        else:
            y_step1=20
        if max_val2-min_val2<6:
            y_step2=1
        elif max_val2-min_val2<11:
            y_step2=2
        elif max_val2-min_val2<31:
            y_step2=5
        else:
            y_step2=10 
        #max(min(5,math.ceil((max_val-min_val)/5)),math.ceil((max_val-min_val)/10))   
        dict_all_plots = {
            1: {
                "metric": metrics[0],
                "y_min": min_val,
                "y_max": max_val,
                "y_step": y_step1,
                "plot_name": f"error_rate_{metrics[0]}.pdf",
            },
            2: {
                "metric": metrics[1],
                "y_min": min_val2,
                "y_max": max_val2,
                "y_step": y_step2,
                "plot_name": f"error_rate_{metrics[1]}.pdf",
            }
        }

        for dict_plot in dict_all_plots.values():

            fig, ax = plt.subplots(figsize=FIGSIZE)
            df2.boxplot(column=dict_plot["metric"], by="error_rate", color=GREEN_2, widths=BAR_WIDTH, ax=ax)

            ax.set_xlabel("Injected error rate", fontsize=LABELS_FONT_SIZE)
            ax.tick_params(axis='x', labelsize=XTICKS_FONT_SIZE)
            ax.set_ylim(dict_plot["y_min"], dict_plot["y_max"])
            ax.set_yticks(np.arange(dict_plot["y_min"], dict_plot["y_max"] + dict_plot["y_step"], dict_plot["y_step"]))
            ax.set_ylabel(dict_plot["metric"]+" (%)", fontsize=LABELS_FONT_SIZE)
            ax.tick_params(axis='y', labelsize=XTICKS_FONT_SIZE)
            ax.set_title("")
            ax.grid(False)
            fig.suptitle("")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","")), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","").replace("pdf","png")), bbox_inches='tight')
            plt.close()
        #plt.show()

def plot_bias_mitigation(plot_dir,csv_path, fairness_metrics):
    df = pd.read_csv(csv_path)
    """
    columns = ["method_id", "method", "Accuracy", "SPD$_{race}$"]
    df = [
        [0, "Clean & No bias mit.", 88, 8],
        [1, "Clean & Bias mit.", 76, 4],
        [2, "No clean & Bias mit.", 75, 0.3],
    ]
    df = pd.DataFrame(df, columns=columns)
    """
    #print(df.columns)


    df2=  df.rename(columns={"precision": "Precision","recall":"Recall","f1":"F1-score","accuracy":"Accuracy","gender_discr_index":"DcI$_{gender}$","gender_spd":"SPD$_{gender}$","gender_avg_odds": "AOD$_{gender}$","gender_eod": "EOD$_{gender}$","gender_disparate_impact": "DI$_{gender}$","age_discr_index":"DcI$_{age}$","age_spd":"SPD$_{age}$","age_avg_odds": "AOD$_{age}$","age_eod": "EOD$_{age}$","age_disparate_impact": "DI$_{age}$","race_discr_index":"DcI$_{race}$","race_spd":"SPD$_{race}$","race_avg_odds": "AOD$_{race}$","race_eod": "EOD$_{race}$","race_disparate_impact": "DI$_{race}$"})
    perf_metrics=[["Accuracy","Recall"],["F1-score","Precision"]]
    #fair_metrics=["SPD$_{gender}$"]

    metrics_list = perf_metrics + fairness_metrics
    #[["Precision", "Recall"],['F1',"Accuracy"],["SPD$_{gender}$","DcI$_{gender}$"],["EOD$_{gender}$","AOD$_{gender}$"],["DI$_{gender}$","F1-score"],["SPD$_{race}$","DcI$_{race}$"],["EOD$_{race}$","AOD$_{race}$"],["DI$_{race}$","F1-score"]]#["SPD$_{age}$","DcI$_{age}$"],["EOD$_{age}$","AOD$_{age}$"],["DI$_{age}$","F1-score"]]
    #,["SPD$_{age}$","DcI$_{age}$"],["EOD$_{age}$","AOD$_{age}$"],["DI$_{age}$","F1-score"]]
    #,["SPD$_{race}$","DcI$_{race}$"],["EOD$_{race}$","AOD$_{race}$"],["DI$_{race}$","F1-score"]]#
    for metrics in metrics_list:
        
        n_last_rounds_agg = 10
        lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds_agg][metrics].mean()
        df2 = df2[df2["exp_seed"]==101]
        df = df2.groupby(["exp_seed","method_name"], as_index=False).apply(lambda_agg)
        FIGSIZE = (5, 4)
        XTICKS_FONT_SIZE = 18
        LABELS_FONT_SIZE = 20
        VALUE_FONT_SIZE = 16
        LEGEND_FONT_SIZE = 13
        BAR_WIDTH = 0.5
        
        GREY = '#CCCCCC'
        VIOLET = '#003366'              # Bleu marine (pour Sel. + hachures)
        GREEN_1  = "#9DCA62"
        GREEN_2  = '#7FB77E'          # Vert sauge clair moderne
        GREEN_3   = '#3A9A75'          # Vert mousse élégant
        GREEN_4 = '#1E6356'          # Vert forêt profond
        BLUE = '#191970'
        ORANGE = "#ffA500"

        HATCH_PATTERN = '//'
        mpl.rcParams['hatch.linewidth'] = 3
        #print(df2.columns,metrics)
        #print("df2",df2)

        #print(df2,metrics[0])
        #print(df2[metrics[0]])
        max_val=math.ceil(df2[metrics[0]].max())
        min_val=min(math.floor(df2[metrics[0]].min()),max_val-5)
        if min_val<0:
            max_val-=min_val
            min_val=0
        if metrics[0] not in ["Precision", "Recall",'F1',"Accuracy"]:
            min_val=0
        small_gap=(max_val-min_val)<30

        if small_gap:
            min_val=math.floor(min_val/5)*5
        else:
            min_val=math.floor(min_val/10)*10
        if max_val>95:
            max_val=105
        else:
            if small_gap:
                max_val=max(math.ceil(max_val/5)*5,max_val*1.1)
            else:
                max_val=max(math.ceil(max_val/10)*10,max_val*1.1)

        max_val2=math.ceil(df2[metrics[1]].max())
        min_val2=min(math.floor(df2[metrics[1]].min()),max_val2-5)
        if min_val2<0:
            max_val2-=min_val2
            min_val2=0
        if metrics[1] not in ["Precision", "Recall",'F1',"Accuracy"]:
            min_val2=0
        small_gap=(max_val2-min_val2)<30
        if small_gap:
            min_val2=math.floor(min_val2/5)*5
        else:
            min_val2=math.floor(min_val2/10)*10
        if max_val2>95:
            max_val2=105
        else:
            if small_gap:
                max_val2=max(math.ceil(max_val2/5)*5,max_val2*1.1)
            else:
                max_val2=max(math.ceil(max_val2/10)*10,max_val2*1.1)
        if max_val-min_val<6:
            y_step1=1
        elif max_val-min_val<11:
            y_step1=2
        elif max_val-min_val<31:
            y_step1=5
        else:
            y_step1=10
        if max_val2-min_val2<6:
            y_step2=1
        elif max_val2-min_val2<11:
            y_step2=2
        elif max_val2-min_val2<31:
            y_step2=5
        else:
            y_step2=10        
        dict_all_plots = {
            1: {
                "metric": metrics[0],
                "y_min": min_val,
                "y_max": max_val,
                "y_step": y_step1,
                "plot_name": f"bias_mitigation_{metrics[0]}.pdf",
            },
            2: {
                "metric": metrics[1],
                "y_min": min_val2,
                "y_max": max_val2,
                "y_step": y_step2,
                "plot_name": f"bias_mitigation_{metrics[1]}.pdf",
            }
        }

        for dict_plot in dict_all_plots.values():
            #print("df",df)
            #print("df",df.columns)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            
            for group_name, df_group in df.groupby("method_name"):
                #print(group_name)
                if group_name == "CustomCentralizedFL all-errors" or group_name == "FedAvg all-errors":
                    bars = ax.bar(0, df_group[dict_plot["metric"]], label="Clean & No bias mit.", color=GREEN_2, width=BAR_WIDTH)
                elif group_name == "ASTRAL all-errors":
                    bars = ax.bar(2, df_group[dict_plot["metric"]], edgecolor=BLUE, hatch=HATCH_PATTERN, label="Clean & Bias mit.", color=GREEN_2, width=BAR_WIDTH)
                elif group_name == "CustomCentralizedFL all-errors-flip"or group_name == "FedAvg all-errors-flip": 
                    bars = ax.bar(1, df_group[dict_plot["metric"]], label="Clean/rev. & No bias mit.", color=ORANGE, width=BAR_WIDTH)
                elif group_name == "ASTRAL all-errors-flip":
                    bars = ax.bar(3, df_group[dict_plot["metric"]], edgecolor=BLUE, hatch=HATCH_PATTERN, label="Clean/rev. & Bias mit.", color=ORANGE, width=BAR_WIDTH)
                else:
                    #print(group_name)
                    assert group_name == "ASTRAL No cleaning"
                    bars = ax.bar(4, df_group[dict_plot["metric"]], label="No clean & Bias mit.", color=BLUE, width=BAR_WIDTH)

                val = df_group[dict_plot["metric"]].iloc[0]
                display_val = f"{val:.1f}" if abs(val) < 1 else f"{round(val)}"
                ax.text(bars[0].get_x() + bars[0].get_width()/2,
                        bars[0].get_height() + (dict_plot["y_max"] - dict_plot["y_min"]) * 0.02,
                        display_val, ha='center', va='bottom',
                        fontsize=VALUE_FONT_SIZE, color='#222', zorder=10)

            #ax.legend(fontsize=LEGEND_FONT_SIZE)
            ax.set_xticklabels([], rotation=0, ha='center', fontsize=12)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_ylim(dict_plot["y_min"], dict_plot["y_max"])
            ax.set_yticks(np.arange(dict_plot["y_min"], dict_plot["y_max"], dict_plot["y_step"]))
            ax.set_ylabel(dict_plot["metric"]+" (%)", fontsize=LABELS_FONT_SIZE)
            ax.tick_params(axis='y', labelsize=XTICKS_FONT_SIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","")), bbox_inches='tight')    
            plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","").replace("pdf","jpg")), bbox_inches='tight')    
            plt.close()
def plot_non_iid(plot_dir,csv_path, fair_metrics,include_x,include_all,include_o):
    df = pd.read_csv(csv_path)
    values = {"dirichlet_alpha":"IID"}
    df=df.fillna(value=values)
    df2=  df.rename(columns={"precision": "Precision","recall":"Recall","f1":"F1-score","accuracy":"Accuracy","gender_discr_index":"DcI$_{gender}$","gender_spd":"SPD$_{gender}$","gender_avg_odds": "AOD$_{gender}$","gender_eod": "EOD$_{gender}$","gender_disparate_impact": "DI$_{gender}$","age_discr_index":"DcI$_{age}$","age_spd":"SPD$_{age}$","age_avg_odds": "AOD$_{age}$","age_eod": "EOD$_{age}$","age_disparate_impact": "DI$_{age}$","race_discr_index":"DcI$_{race}$","race_spd":"SPD$_{race}$","race_avg_odds": "AOD$_{race}$","race_eod": "EOD$_{race}$","race_disparate_impact": "DI$_{race}$"})
    n_last_rounds_agg=10
    #metrics=["Accuracy","SPD$_{gender}$"]
    perf_metrics=["Accuracy","Recall","F1-score","Precision"]
    #fair_metrics=["SPD$_{gender}$"]
    for perf_metric in perf_metrics:
        for fairness_metric in fair_metrics:
            metrics=[perf_metric,fairness_metric]
            lambda_agg = lambda df_lambda: df_lambda.loc[df_lambda["round"] > df_lambda["round"].max() - n_last_rounds_agg][metrics].mean()
            df2 = df2[df2["exp_seed"]==16]
            df = df2.groupby(["dirichlet_alpha","data_cleaning"], as_index=False).apply(lambda_agg)
            for metric in metrics:
                """
                if "DI$_" not in metric:
                    df[metric] = df[metric]*100
                else:
                    df[metric].where(df[metric] >= 1 / df[metric], 1 / df[metric])
                    #df[metric] = max(df[metric],1/df[metric])
                """
                df[metric] = df[metric].abs()
            """
            df = [
                ["Clean", "0.01", 81, 27.30],
                ["Clean", "0.05", 84.60, 23.60],
                ["Clean", "0.1", 85.10, 12.90],
                ["Clean", "IID", 88.30, 1.40],

                ["No clean", "0.01", 80.30, 18.20],
                ["No clean", "0.05", 82.50, 9.30],
                ["No clean", "0.1", 83.10, 0.40],
                ["No clean", "IID", 84.30, 2.00],

                ["FL clean", "0.01", 77.70, 10.20],
                ["FL clean", "0.05", 80.90, 23.44],
                ["FL clean", "0.1", 80.20, 15.71],
                ["FL clean", "IID", 83.20, 14.29],
            ]
            df = pd.DataFrame(df, columns=columns)
            """

            FIGSIZE = (10, 6)
            XTICKS_FONT_SIZE = 18
            LABELS_FONT_SIZE = 20
            LEGEND_FONT_SIZE = 13
            MARKER_SIZE = 80

            color_mapping = {
                "0.01": "#941651",
                "0.05": "#009051",
                "0.1": "#011893",
                "IID": "#FF9300",
            }

            max_val=math.ceil(df[metrics[0]].max())
            min_val=min(math.floor(df[metrics[0]].min()),max_val-5)
            small_val=max_val<30
            if small_val:
                min_val=math.floor(min_val/5)*5
            else:
                min_val=math.floor(min_val/10)*10
     

            if min_val<0:
                max_val-=min_val
                min_val=0
            if metrics[0] not in ["Precision", "Recall",'F1',"Accuracy"]:
                min_val=0
            if small_val:
                max_val2=math.ceil(max_val/5)*5
            else:
                max_val=math.ceil(max_val/10)*10   
            max_val2=math.ceil(df[metrics[1]].max())
            min_val2=min(math.floor(df[metrics[1]].min()),max_val2-5)
            small_val=max_val2<30
            if small_val:
                min_val2=math.floor(min_val2/5)*5
            else:
                min_val2=math.floor(min_val2/10)*10   
            if min_val2<0:
                max_val2-=min_val2
                min_val2=0
            if metrics[1] not in ["Precision", "Recall",'F1',"Accuracy"]:
                min_val2=0
            if small_val:
                max_val2=math.ceil(max_val2/5)*5
            else:
                max_val2=math.ceil(max_val2/10)*10   

            if max_val2<6:
                y_step=1
            elif max_val2<11:
                y_step=2
            elif max_val2<50:
                y_step=5
            else:
                y_step=10
            if max_val-min_val<6:
                x_step=1
            elif max_val-min_val<11:
                x_step=2
            elif max_val-min_val<31:
                x_step=5
            else:
                x_step=10
            dict_all_plots = {
                1: {
                    "y": metrics[1],
                    "y_min": 0,
                    "y_max": max_val2,
                    "y_step": y_step,
                    "x": metrics[0],
                    "x_min": min_val,
                    "x_max": max_val,
                    "x_step": x_step,
                    "plot_name": f"non_iid_{metrics[0]}_{metrics[1]}.pdf",
                }
            }

            for dict_plot in dict_all_plots.values():

                fig, ax = plt.subplots(figsize=FIGSIZE)
                for group_name, df_group in df.groupby(["dirichlet_alpha", "data_cleaning"]):
                    #print(df_group.values[0][0],",",df_group.values[0][1],",",df_group.values[0][2],",",df_group.values[0][3])
                    if (group_name[0]==0.01 or group_name[0]=="IID") or include_all:
                        if group_name[1] == "multi_error_types" or group_name[1] == "all-errors":
                            if include_o:
                                df_group.plot.scatter(x=dict_plot["x"], y=dict_plot["y"], ax=ax, marker="^", linewidths=2, c='w', edgecolors=color_mapping[str(group_name[0])], s=MARKER_SIZE, label="{} w/ cleaning".format(str(group_name[0])))
                        elif group_name[1] == "multi_error_types_FL" or group_name[1] == "all-errors-FL":
                            if include_o:
                                df_group.plot.scatter(x=dict_plot["x"], y=dict_plot["y"], ax=ax, marker="s", linewidths=2, c='w', edgecolors=color_mapping[str(group_name[0])], s=MARKER_SIZE, label="{} w/ cleaning+".format(str(group_name[0])))
                        elif group_name[1] == "multi_error_types_flip" or group_name[1]=="all-errors-flip":
                            if include_x:
                                df_group.plot.scatter(x=dict_plot["x"], y=dict_plot["y"], ax=ax, marker="*", linewidths=2, c='w', edgecolors=color_mapping[str(group_name[0])], s=MARKER_SIZE, label="{} w/ cleaning flipped".format(str(group_name[0])))
                        elif group_name[1] == "multi_error_types_FL_flip" or group_name[1]=="all-errors-FL-flip":
                            if include_x:
                                df_group.plot.scatter(x=dict_plot["x"], y=dict_plot["y"], ax=ax, marker="P", linewidths=2, c='w', edgecolors=color_mapping[str(group_name[0])], s=MARKER_SIZE, label="{} w/ cleaning+ flipped".format(str(group_name[0])))
                        else:
                            df_group.plot.scatter(x=dict_plot["x"], y=dict_plot["y"], ax=ax, marker="o", c=color_mapping[str(group_name[0])], s=MARKER_SIZE, label="{} w/o cleaning".format(str(group_name[0])))


                MARKER_SIZE=8
                """
                patch_1 = mpatches.Patch(color=color_mapping["0.01"], edgecolor='black', label="non-IID 0.01")
                patch_2 = mpatches.Patch(color=color_mapping["0.05"], edgecolor='black', label="non-IID 0.05")
                patch_3 = mpatches.Patch(color=color_mapping["0.1"], edgecolor='black', label="non-IID 0.1")
                patch_4 = mpatches.Patch(color=color_mapping["IID"], edgecolor='black', label="IID")
                marker_1 = Line2D([0], [0], color='w', marker='o', markerfacecolor='black', label="w/o cleaning", markersize=MARKER_SIZE)
                marker_2 = Line2D([0], [0], color='w', marker='^', markerfacecolor='w', markeredgecolor='black', label="w/ cleaning", markersize=MARKER_SIZE)
                marker_3 = Line2D([0], [0], color='w', marker='*', markerfacecolor='w', markeredgecolor='black', label="w/ cleaning rev.", markersize=MARKER_SIZE)
                marker_4 = Line2D([0], [0], color='w', marker='s', markerfacecolor='w', markeredgecolor='black', label="w/ cleaning+", markersize=MARKER_SIZE)
                marker_5 = Line2D([0], [0], color='w', marker='P', markerfacecolor='w', markeredgecolor='black', label="w/ cleaning+ rev.", markersize=MARKER_SIZE)
                legend = [patch_1, patch_2, patch_3, patch_4, marker_1, marker_2, marker_3,marker_4,marker_5]
                ax.legend(handles=legend, fontsize=LEGEND_FONT_SIZE)
                """
                ax.get_legend().remove()
                ax.grid()
                ax.set_ylim(dict_plot["y_min"], dict_plot["y_max"])
                ax.set_yticks(np.arange(dict_plot["y_min"], dict_plot["y_max"] + dict_plot["y_step"], dict_plot["y_step"]))
                ax.set_ylabel(dict_plot["y"]+" (%)", fontsize=LABELS_FONT_SIZE)
                ax.tick_params(axis='y', labelsize=XTICKS_FONT_SIZE)
                ax.set_xlim(dict_plot["x_min"], dict_plot["x_max"])
                ax.set_xticks(np.arange(dict_plot["x_min"], dict_plot["x_max"] + dict_plot["x_step"], dict_plot["x_step"]))
                ax.set_xlabel(dict_plot["x"]+" (%)", fontsize=LABELS_FONT_SIZE)
                ax.tick_params(axis='x', labelsize=XTICKS_FONT_SIZE)
                #plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","")), bbox_inches='tight')
                plt.savefig(os.path.join(plot_dir, dict_plot["plot_name"].replace("$","").replace("{","").replace("}","").replace("pdf","png")), bbox_inches='tight')
                plt.close()

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--dataset', required=True, help='comma seperated list: name of the dataset being used')
    parser.add_argument('--experiment',required=True, help='comma seperated list: name of experiment being tested')
    args = parser.parse_args()
    main(args.dataset,args.experiment)
