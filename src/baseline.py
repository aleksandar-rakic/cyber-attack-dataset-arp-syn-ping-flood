"""Baseline for the Cyber Attack Dataset v3.

Trains two models on two feature sets and scores them on the chronological
test split and, for comparison, on a random stratified split. Every fitted
step (imputer, scaler, model) sees the training rows only.

Run:  python src/baseline.py --data network_attack_traffic.csv
      python src/baseline.py --data data/sample_25k.csv     (quick check)
"""
import argparse
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["Normal", "ARP_Spoofing", "ARP_Storm", "PING_Flood", "SYN_Flood"]
FLAGS = ["tcp_flag_fin", "tcp_flag_syn", "tcp_flag_rst", "tcp_flag_psh", "tcp_flag_ack"]
HEADER = ["frame_time_delta", "arp_opcode", "icmp_type_code", "tcp_seq", "tcp_hdr_len", "data_len"] + FLAGS
FEATURE_SETS = {
    # What a packet header alone says. The honest set.
    "header_only": HEADER,
    # Adds the capture position and the device roles, which give the answer away.
    "with_shortcuts": HEADER + ["frame_number", "arp_src_device", "arp_dst_device"],
}


def encode(df):
    """Numeric matrix: categoricals become integer codes, missing stays NaN."""
    x = df.copy()
    x["arp_opcode"] = x["arp_opcode"].map({"request": 1, "reply": 2})
    for col in x.columns.intersection(["arp_src_device", "arp_dst_device"]):
        x[col] = x[col].map({"attacker": 0, "victim": 1, "unresolved": 2, "other_1": 3,
                             "other_2": 4, "other_3": 5, "other_4": 6})
    return x.astype(float)


def models():
    return {
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000)),
    }


def score(train, test, cols):
    rows = {}
    for name, model in models().items():
        model.fit(encode(train[cols]), train["label"])
        pred = model.predict(encode(test[cols]))
        recall = recall_score(test["label"], pred, average=None, labels=range(5))
        rows[name] = {"macro_f1": round(float(f1_score(test["label"], pred, average="macro")), 4),
                      "recall": {LABELS[i]: round(float(r), 4) for i, r in enumerate(recall)}}
    return rows


parser = argparse.ArgumentParser(description="Baseline models for the v3 dataset")
parser.add_argument("--data", default="network_attack_traffic.csv", help="full table or the 25k sample")
parser.add_argument("--out", default="baseline_results.json")
args = parser.parse_args()

full = pd.read_csv(args.data)
chrono = (full[full.split == "train"], full[full.split == "test"])
random = train_test_split(full, test_size=0.2, stratify=full["label"], random_state=42)

results = {}
for split_name, (train, test) in {"chronological": chrono, "random_stratified": random}.items():
    for set_name, cols in FEATURE_SETS.items():
        results[f"{split_name} / {set_name}"] = score(train, test, cols)

with open(args.out, "w") as fh:
    json.dump(results, fh, indent=2)
for key, rows in results.items():
    for model, r in rows.items():
        print(f"{key:38s} {model:20s} macro-F1 {r['macro_f1']:.4f}  recall {r['recall']}")
