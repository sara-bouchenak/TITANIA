import json
import argparse


def main(path_to_results_1, path_to_results_2):

    with open(path_to_results_1, "r") as f:
        metrics_dict_1 = json.load(f)
    
    with open(path_to_results_2, "r") as f:
        metrics_dict_2 = json.load(f)

    if metrics_dict_1["perf_global"] == metrics_dict_2["perf_global"]:
        print("The results are the same!")
    else:
        print("The results are not the same!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_json_1', required=True)
    parser.add_argument('--path_to_json_2',required=True)
    args = parser.parse_args()
    main(args.path_to_json_1, args.path_to_json_2)
