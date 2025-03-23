import os
import glob
import re
import pandas as pd

TRAIN_RESULTS_DIR = "outputs"
OUTPUT_CSV = "csvs/results_full_sweep.csv"

def process_file(filename):
    with open(filename, "r") as f:
        info = {}
        info["dataset"] = os.path.basename(filename).replace(".txt", "")
        line = "\n"
        while line !="":
            line = f.readline()
            best_hyper_match = re.match("Best hyperparameters:\s?(\{.*\})", line)
            if best_hyper_match:
                hyperparam = eval(best_hyper_match.group(1))
                for (k, v) in hyperparam.items():
                    info[k.lower()] = v
                if "validation accuracy" in info:
                    info["validation_accuracy"] = round(info["validation accuracy"] * 100, 2)
                    del info["validation accuracy"]
                continue
            
            acc_match = re.match("Final accuracy with best hyperparameters:\s?([0-9.]*)", line)
            if acc_match:
                info['test_accuracy'] = acc_match.group(1)
                continue

            tr_time_match = re.match("Total training time:\s?([0-9.e-]*)", line)
            if tr_time_match:
                info['training_time'] = tr_time_match.group(1)
                continue

            inf_time_match = re.match("Total inference time:\s?([0-9.e-]*)", line)
            if inf_time_match:
                info['inference_time'] = inf_time_match.group(1)
                continue

            if "MEMORY REPORT" in line:
                break
        
        peak_memory = None
        while line != "":
            line = f.readline()
            match = re.match("\s*[0-9]*\s*([0-9.]*) MiB\s*", line)
            if match:
                mem = float(match.group(1))
                if peak_memory is None or mem > peak_memory:
                    peak_memory = mem
        if peak_memory:
            info['peak_memory'] = peak_memory
    return info


def main():
    files = glob.glob(f"{TRAIN_RESULTS_DIR}/*")

    df = pd.DataFrame()
    for filename in files:
        info = process_file(filename)
        if len(info) == 1:
            continue
        for k in info:
            info[k] = [info[k]]
        df = pd.concat((df, pd.DataFrame(info)), axis=0)

    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
    