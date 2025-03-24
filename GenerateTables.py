import os
import glob
import re
import pandas as pd
from dataclasses import dataclass

##### TABLE CONFIGURATION: CAN EDIT #####
DIR = "outputs"
DNABERT_CSV = "csvs/dnabert_accuracies.csv"

@dataclass
class TableColInfo:
    category: str
    dataframe_keys: list[str]
    col_names: list[str]

@dataclass
class TableRowInfo:
    dataframe_key: str
    latex: str
    
PROM_TYPES = ["all", "tata", "notata"]
TABLE_COLS = [
    [
        TableColInfo("mouse", [f"mouse{i}" for i in range(5)], [f"{i}" for i in range(5)]),
        TableColInfo("tf", [f"tf{i}" for i in range(5)], [f"{i}" for i in range(5)])
    ],
    [
        TableColInfo("prom core", [f"prom_core_{tpe}" for tpe in PROM_TYPES], PROM_TYPES),
        TableColInfo("prom 300", [f"prom_300_{tpe}" for tpe in PROM_TYPES], PROM_TYPES),
        TableColInfo("", ["splice"], ["splice"]),
        TableColInfo("EMP H4", ["H4", "H4ac"], ["H4", "ac"])
    ],
    [
        TableColInfo("EMP H3", ['H3', 'H3K14ac', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me3', 'H3K9ac'], 
                     ['H3', 'K14ac', 'K36me3', 'K4me1', 'K4me2', 'K4me3', 'K79me3', 'K9ac']),
        TableColInfo("", ["covid"], ["COVID"])
    ]
]

TABLE_ROWS = [
    TableRowInfo("DNABERT-2", "DNABERT-2"), TableRowInfo("DNABERT-2-*", "DNABERT-2$\diamond$"), TableRowInfo("test_accuracy", "LZ78")
]
INITIAL_TABS = 2
##########################################


def process_file(filename):
    with open(filename, "r") as f:
        info = {}
        info["dataset"] = os.path.basename(filename).replace(".txt", "")
        line = "\n"
        while line !="":
            line = f.readline()
            acc_match = re.match("Final accuracy with best hyperparameters:\s?([0-9.]*)", line)
            if acc_match:
                info['test_accuracy'] = acc_match.group(1)
                continue
    return info


def main():
    files = glob.glob(f"{DIR}/*")

    df = pd.DataFrame()
    for filename in files:
        info = process_file(filename)
        if len(info) == 1:
            continue
        for k in info:
            info[k] = [info[k]]
        df = pd.concat((df, pd.DataFrame(info)), axis=0, ignore_index=True)

    df = df[["dataset", "test_accuracy"]].set_index("dataset")
    bert_df = pd.read_csv(DNABERT_CSV)[["dataset", "DNABERT-2", "DNABERT-2-*"]].set_index("dataset")
    df = df.join(other=bert_df)
    df["test_accuracy"] = pd.to_numeric(df["test_accuracy"])
    df["best"] = list(df.idxmax(axis=1))

    indent = " " * (INITIAL_TABS * 4)
    for (table_num, table_col) in enumerate(TABLE_COLS):
        n_cols = [len(col_info.col_names) for col_info in table_col]
        print(indent + "\\begin{tabular}{c|" + "|".join(["c" * n for n in n_cols]) + "}")
        if table_num == 0:
            print(indent + "    \\toprule")
        
        col_header_data = [(len(col_info.col_names), ("|" if i != len(table_col) - 1 else ""), col_info.category) for (i, col_info) in enumerate(table_col)]
        print(indent + "   " + " & ".join([""] + ["\\multicolumn{%d}{c%s}{\\textbf{%s}}" % data for data in col_header_data]) + " \\\\")
        col_names = sum([col_info.col_names for col_info in table_col], start=[])
        col_keys = sum([col_info.dataframe_keys for col_info in table_col], start=[])
        print(indent + "   " + " & ".join([""] + ["\\textbf{%s}" % name for name in col_names]) + " \\\\")
        print(indent + "    \\midrule")
        for row_info in TABLE_ROWS:
            print(indent + "    \\textbf{%s} & " % row_info.latex, end="")
            values = list(df[row_info.dataframe_key][col_keys])
            bolded = list(df["best"][col_keys] == row_info.dataframe_key)
            print(indent + " & ".join([str(round(val, 2)) if not bold else ("\\textbf{%0.2f}" % val)  for (val, bold) in zip(values, bolded)]) + " \\\\")
        if table_num != len(TABLE_COLS) - 1:
            print(indent + "    \\midrule")
        else:
            print(indent + "    \\bottomrule")
        print(indent + "\\end{tabular}")
        if table_num != len(TABLE_COLS) - 1:
            print(indent + "\\vskip1em")

if __name__ == "__main__":
    main()