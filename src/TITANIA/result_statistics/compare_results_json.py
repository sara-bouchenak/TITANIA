import json
import argparse


def main(path_to_results_1, path_to_results_2, max_print_difference=10):

    with open(path_to_results_1, "r") as f:
        metrics_dict_1 = json.load(f)
    
    with open(path_to_results_2, "r") as f:
        metrics_dict_2 = json.load(f)

    diff_list = []
    n_diff = 0 
    for round in metrics_dict_1["perf_global"].keys():
        for metric in metrics_dict_1["perf_global"][round].keys():
            metric_1 = metrics_dict_1["perf_global"][round][metric]
            metric_2 = metrics_dict_2["perf_global"][round][metric]
            if metric_1 != metric_2:
                if n_diff < max_print_difference:
                    diff_list.append([round, metric, metric_1, metric_2])
                n_diff += 1

    print(f"Their are a total of {n_diff} difference(s)!")
    if min(n_diff, max_print_difference) > 0 :
        print(f"Here are {min(n_diff, max_print_difference)} example(s):")
        for diff in diff_list:
            print(f"At round {diff[0]} for {diff[1]}: {diff[2]} compared to {diff[3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_json_1', required=True)
    parser.add_argument('--path_to_json_2',required=True)
    args = parser.parse_args()
    main(args.path_to_json_1, args.path_to_json_2)
