import os
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


rnd_seed = 42


def save_tabular_jsonl(data, output_dir, target_col=None):
    for split, df in data.items():
        save_file = os.path.join(output_dir, f"{split}.jsonl")
        save_csv_file = os.path.join(output_dir, f"{split}.csv")

        df_no_id = df.drop(columns=["id"], inplace=False)
        df_no_id.to_csv(save_csv_file, index=False)

        with open(save_file, "w") as wf:
            for _, row in df.iterrows():
                row_dict = {k: v for k, v in row.to_dict().items() if k != "id"}
                res = {
                    "id": int(row["id"]),
                    "text": json.dumps(row_dict, ensure_ascii=False),
                }
                if target_col is not None:
                    res["target_col"] = target_col
                    res["target"] = row_dict[target_col]
                wf.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"{len(df)} samples saved to {save_file}")


def mirror_to_synthetic(dataset_name, data_dir, synthetic_dir):
    # Mirror the prepared tabular dataset to data/synthetic/<dataset>/
    if synthetic_dir is None:
        return

    syn_dir = os.path.join(synthetic_dir, dataset_name)
    os.makedirs(syn_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    info_json = os.path.join(data_dir, "info.json")

    if os.path.exists(train_csv):
        shutil.copyfile(train_csv, os.path.join(syn_dir, "real.csv"))
    if os.path.exists(test_csv):
        shutil.copyfile(test_csv, os.path.join(syn_dir, "test.csv"))
    if os.path.exists(info_json):
        shutil.copyfile(info_json, os.path.join(syn_dir, "info.json"))

    print(f"[{dataset_name}] also mirrored to {syn_dir}")


def save_dataset_info(data_df, task, task_type, names, num_col, cat_col, tgt_col, output_dir):
    num_col_idx = sorted([names.index(x) for x in num_col])
    cat_col_idx = sorted([names.index(x) for x in cat_col])
    tgt_col_idx = sorted([names.index(x) for x in tgt_col])

    col_info = dict()
    for col in names:
        if col in num_col or (col in tgt_col and task_type == "regression"):
            col_info[col] = {
                "type": "float",
                "min": float(data_df[col].astype(float).min()),
                "max": float(data_df[col].astype(float).max()),
                "mean": float(data_df[col].astype(float).mean()),
                "median": float(data_df[col].astype(float).median()),
                "std": float(data_df[col].astype(float).std()),
            }
        else:
            col_info[col] = {
                "type": "category",
                "candidates": [str(x) for x in data_df[col].unique()],
            }

    # add metadata
    metadata = {"columns": {}}
    for i, col in enumerate(names):
        if col in num_col or (col in tgt_col and task_type == "regression"):
            metadata["columns"][str(i)] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        else:
            metadata["columns"][str(i)] = {"sdtype": "categorical"}

    data_info = {
        "name": task,
        "task_type": task_type,
        "column_names": names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": tgt_col_idx,
        "column_info": col_info,
        "metadata": metadata,
    }
    with open(os.path.join(output_dir, "info.json"), "w") as wf:
        json.dump(data_info, wf, indent=4, ensure_ascii=False)


def save_dataset_info_wtext(
    data_df,
    dataset_name,
    task_type,
    names,
    num_col,
    cat_col,
    text_col,
    tgt_col,
    output_dir,
):
    num_col_idx = sorted([names.index(x) for x in num_col if x not in tgt_col])
    cat_col_idx = sorted([names.index(x) for x in cat_col if x not in tgt_col])
    text_col_idx = sorted([names.index(x) for x in text_col])
    tgt_col_idx = sorted([names.index(x) for x in tgt_col])

    col_info = {}

    for col in names:
        if col in num_col:
            s = data_df[col].astype(float)
            col_info[col] = {
                "type": "float",
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
            }

        elif col in text_col:
            s = data_df[col].astype(str).fillna("")
            wc = s.apply(lambda x: len(x.split()))
            col_info[col] = {
                "type": "text",
                "min": int(wc.min()),
                "max": int(wc.max()),
                "mean": float(wc.mean()),
                "median": float(wc.median()),
                "std": float(wc.std()),
            }

        else:
            col_info[col] = {
                "type": "category",
                "candidates": [str(x) for x in data_df[col].dropna().unique()],
            }

    # add metadata
    metadata = {"columns": {}}
    for i, col in enumerate(names):
        if col in num_col:
            metadata["columns"][str(i)] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        elif col in text_col:
            metadata["columns"][str(i)] = {"sdtype": "text"}
        else:
            metadata["columns"][str(i)] = {"sdtype": "categorical"}

    data_info = {
        "name": dataset_name,
        "task_type": task_type,
        "column_names": names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "text_col_idx": text_col_idx,
        "target_col_idx": tgt_col_idx,
        "column_info": col_info,
        "metadata": metadata,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "info.json"), "w") as wf:
        json.dump(data_info, wf, indent=4, ensure_ascii=False)

    print(f"[save_dataset_info_wtext] info saved to {os.path.join(output_dir, 'info.json')}")


def process_math_latex(base_dir, synthetic_dir=None):
    print("########## math_latex ##########")
    dataset_name = "math_latex"
    data_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_ori.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_ori.csv"))

    num_col = ["x1", "x2"]
    cat_col = ["operation_x1", "operation_x2", "operation_between"]
    text_col = ["latex_expression"]

    target_col = "latex_expression"

    col_order = num_col + cat_col + text_col
    train_df = train_df[col_order]
    valid_df = valid_df[col_order]
    names = col_order

    train_df = train_df.reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df))

    valid_df = valid_df.reset_index(drop=True)
    valid_df["id"] = np.arange(len(valid_df))

    # math_latex has no held-out test split; reuse valid as test so the
    # downstream pipeline (real.csv / test.csv layout) still works.
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": valid_df,
    }

    save_tabular_jsonl(data, data_dir, target_col=target_col)
    full_df = train_df[names]

    save_dataset_info_wtext(full_df, dataset_name, "consistence_check", names, num_col, cat_col, text_col, [target_col], data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_rel_amazon(base_dir, synthetic_dir=None):
    print("########## rel_amazon ##########")
    dataset_name = "rel_amazon"
    data_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_ori.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_ori.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_ori.csv"))

    num_col = ["price", "review_time"]
    cat_col = ["rating", "category", "verified"]
    text_col = ["brand", "title", "description", "review_text", "summary"]

    target_col = "rating"

    col_order = num_col + cat_col + text_col
    train_df = train_df[col_order]
    valid_df = valid_df[col_order]
    test_df = test_df[col_order]
    names = col_order

    train_df = train_df.reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df))

    valid_df = valid_df.reset_index(drop=True)
    valid_df["id"] = np.arange(len(valid_df))

    test_df = test_df.reset_index(drop=True)
    test_df["id"] = np.arange(len(test_df))

    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    save_tabular_jsonl(data, data_dir, target_col=target_col)
    full_df = train_df[names]

    save_dataset_info_wtext(full_df, dataset_name, "mulclass", names, num_col, cat_col, text_col, [target_col], data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_rel_arxiv(base_dir, synthetic_dir=None):
    print("########## rel_arxiv ##########")
    dataset_name = "rel_arxiv"
    data_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_ori.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_ori.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_ori.csv"))

    # numerical: time since first submission, calendar year, whitespace word counts
    num_col = [
        "days_since_first_submit",
        "paper_year",
        "title_word_count",
        "abstract_word_count",
    ]
    # categorical: only the target (Primary Category)
    cat_col = ["Category"]
    # free-text fields
    text_col = ["Title", "arXiv_Code", "Abstract"]

    target_col = "Category"

    col_order = num_col + cat_col + text_col
    train_df = train_df[col_order]
    valid_df = valid_df[col_order]
    test_df = test_df[col_order]

    # Category candidates in info.json come from the *training* split,
    # which mirrors how rel_amazon is set up.
    names = col_order

    train_df = train_df.reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df))

    valid_df = valid_df.reset_index(drop=True)
    valid_df["id"] = np.arange(len(valid_df))

    test_df = test_df.reset_index(drop=True)
    test_df["id"] = np.arange(len(test_df))

    data = {
        "train": train_df,
        "valid": valid_df,
        "test":  test_df,
    }

    save_tabular_jsonl(data, data_dir, target_col=target_col)
    full_df = train_df[names]

    save_dataset_info_wtext(full_df, dataset_name, "mulclass", names,
                            num_col, cat_col, text_col, [target_col], data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_biography(base_dir, synthetic_dir=None):
    print("########## biography ##########")
    dataset_name = "biography"
    data_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_ori.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_ori.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_ori.csv"))

    num_col = ["age", "salary"]
    cat_col = ["sex", "birth_state", "college", "degree", "occupation"]
    text_col = ["biography"]

    target_col = "salary"

    col_order = num_col + cat_col + text_col
    train_df = train_df[col_order]
    valid_df = valid_df[col_order]
    test_df = test_df[col_order]
    names = col_order

    train_df = train_df.reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df))

    valid_df = valid_df.reset_index(drop=True)
    valid_df["id"] = np.arange(len(valid_df))

    test_df = test_df.reset_index(drop=True)
    test_df["id"] = np.arange(len(test_df))

    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    save_tabular_jsonl(data, data_dir, target_col=target_col)
    full_df = train_df[names]

    save_dataset_info_wtext(full_df, dataset_name, "regression", names, num_col, cat_col, text_col, [target_col], data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_adult(base_dir, synthetic_dir=None):
    print("########## adult ##########")
    dataset_name = "adult"
    data_dir = os.path.join(base_dir, dataset_name)
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    data_df = pd.read_csv(os.path.join(data_dir, "adult.data"), names=names, skipinitialspace=True)
    test_df = pd.read_csv(os.path.join(data_dir, "adult.test"), names=names, skipinitialspace=True)
    data_df["id"] = list(range(len(data_df))) # keep origin index
    test_df["id"] = list(range(len(data_df), len(data_df)+len(test_df)))
    test_df["income"] = test_df["income"].str.rstrip(".")
    train_df, valid_df = train_test_split(data_df, test_size=0.1, random_state=rnd_seed)

    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="income")

    # info
    full_df = pd.concat([data_df, test_df])
    num_col = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_col = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    tgt_col = ["income", ]
    save_dataset_info(full_df, dataset_name, "binclass", names, num_col, cat_col, tgt_col, data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("###########################")


def process_beijing(base_dir, synthetic_dir=None):
    print("########## beijing ##########")
    dataset_name = "beijing"
    data_dir = os.path.join(base_dir, dataset_name)
    data_df = pd.read_csv(os.path.join(data_dir, "PRSA_data_2010.1.1-2014.12.31.csv"), header=0)
    data_df.rename(columns={"No": "id"}, inplace=True)

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": pd.concat([train_df, valid_df], ignore_index=True),
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="pm2.5")

    # info
    names = ["year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    num_col = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    cat_col = ["year", "month", "day", "hour", "cbwd"]
    tgt_col = ["pm2.5", ]
    save_dataset_info(data_df, dataset_name, "regression", names, num_col, cat_col, tgt_col, data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_default(base_dir, synthetic_dir=None):
    print("########## default ##########")
    dataset_name = "default"
    data_dir = os.path.join(base_dir, dataset_name)
    data_df = pd.read_excel(os.path.join(data_dir, "default of credit card clients.xls"), skiprows=[0], dtype=int)
    data_df.rename(columns={"ID": "id"}, inplace=True)

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": pd.concat([train_df, valid_df], ignore_index=True),
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="default payment next month")

    # info
    names = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "default payment next month"]
    num_col = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    cat_col = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    tgt_col = ["default payment next month", ]
    save_dataset_info(data_df, dataset_name, "binclass", names, num_col, cat_col, tgt_col, data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def process_magic(base_dir, synthetic_dir=None):
    print("########## magic ##########")
    dataset_name = "magic"
    data_dir = os.path.join(base_dir, dataset_name)
    names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    data_df = pd.read_csv(os.path.join(data_dir, "magic04.data"), names=names)
    data_df["id"] = list(range(len(data_df)))

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": pd.concat([train_df, valid_df], ignore_index=True),
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="class")

    # info
    num_col = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
    cat_col = []
    tgt_col = ["class", ]
    save_dataset_info(data_df, dataset_name, "binclass", names, num_col, cat_col, tgt_col, data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("###########################")


def process_shoppers(base_dir, synthetic_dir=None):
    print("########## shoppers ##########")
    dataset_name = "shoppers"
    data_dir = os.path.join(base_dir, dataset_name)
    data_df = pd.read_csv(os.path.join(data_dir, "online_shoppers_intention.csv"), header=0)
    data_df["id"] = list(range(len(data_df)))

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": pd.concat([train_df, valid_df], ignore_index=True),
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="Revenue")

    # info
    names = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]
    num_col = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    cat_col = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    tgt_col = ["Revenue", ]
    save_dataset_info(data_df, dataset_name, "binclass", names, num_col, cat_col, tgt_col, data_dir)
    mirror_to_synthetic(dataset_name, data_dir, synthetic_dir)
    print("#############################")


def run():
    base_dir = "./data/tabular"
    synthetic_dir = "./data/synthetic"

    ### Tabular datasets Preprocessing ###
    # All datasets have been preprocessed and save in data/tabular/dataset_name/
    ### Tabular datasets Preprocessing ###

    # Plain tabular datasets
    # process_adult(base_dir, synthetic_dir=synthetic_dir)
    # process_beijing(base_dir, synthetic_dir=synthetic_dir)
    # process_default(base_dir, synthetic_dir=synthetic_dir)
    # process_magic(base_dir, synthetic_dir=synthetic_dir)
    # process_shoppers(base_dir, synthetic_dir=synthetic_dir)

    # Datasets with free-form text columns
    # process_math_latex(base_dir, synthetic_dir=synthetic_dir)
    # process_rel_amazon(base_dir, synthetic_dir=synthetic_dir)
    # process_biography(base_dir, synthetic_dir=synthetic_dir)
    # process_rel_arxiv(base_dir, synthetic_dir=synthetic_dir)


if __name__ == "__main__":
    run()
