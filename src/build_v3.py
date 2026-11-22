"""Build release v3 of the Cyber Attack Dataset (ARP, SYN, PING flood).

Versions 1 and 2 published only globally z-score standardized values. This
script restores the original units by inverting that scaling, and checks the
result against the raw Wireshark exports of the four attack captures.

Inputs (--input-dir)
  dataset.csv                       v1 standardized table (439,171 rows, 14 features + label)
  raw_sources/<attack>_20240821.csv raw Wireshark exports of the attack packets
  device_roles.json                 {"attacker": "<name as Wireshark shows it>", "victim": "<name>"}

The inputs are not in this repository: the raw exports carry device
identifiers. The outputs are published on Kaggle (DOI 10.34740/kaggle/ds/5620663).

Outputs (--output-dir)
  network_attack_traffic.csv, train.csv, test.csv, sample_25k.csv,
  raw_attack_captures/*.csv, label_mapping.csv, recovery_report.json,
  dataset_qc_report.json

Run:  python src/build_v3.py --input-dir <inputs> --output-dir release_v3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", default="release_v3")
args = parser.parse_args()
HERE = args.input_dir
OUT = args.output_dir
RAW = {1: "arp_spoof", 2: "arp_storm", 3: "smurf", 4: "syn_flood"}
LABELS = {0: "Normal", 1: "ARP_Spoofing", 2: "ARP_Storm", 3: "PING_Flood", 4: "SYN_Flood"}
RAW_OUT_NAME = {1: "arp_spoofing", 2: "arp_storm", 3: "ping_flood", 4: "syn_flood"}
FLAGS = ["fin", "syn", "rst", "psh", "ack"]
with open(os.path.join(HERE, "device_roles.json")) as fh:
    ROLES = json.load(fh)
ATTACKER_MAC = ROLES["attacker"]
VICTIM_MAC = ROLES["victim"]
report = {"method": "affine inversion of StandardScaler, anchored on raw attack captures", "columns": {}}


def nullable_int(values, missing):
    """Integer column with real missing values where `missing` is True."""
    return pd.Series(np.where(missing, 0, np.round(values)).astype(np.int64)).astype("Int64").mask(missing).values


def levels(s):
    return np.sort(s.unique())


def fit_affine(zv, xv):
    """Least-squares x = a*z + b; returns a, b and the worst absolute error."""
    a, b = np.polyfit(zv, xv, 1)
    return a, b, float(np.max(np.abs(a * zv + b - xv)))


z = pd.read_csv(os.path.join(HERE, "dataset.csv"))
raw = {k: pd.read_csv(os.path.join(HERE, "raw_sources", f"{n}_20240821.csv"),
                      encoding="ISO-8859-1", low_memory=False) for k, n in RAW.items()}
for k, d in raw.items():
    assert len(d) == (z.label == k).sum(), f"row count mismatch for label {k}"

# ---- frame.number: order-preserving match inside each attack class gives exact pairs
zs, xs = [], []
for k, d in raw.items():
    zs.append(np.sort(z.loc[z.label == k, "frame.number"].values))
    xs.append(np.sort(d["frame.number"].values.astype(float)))
a, b, err = fit_affine(np.concatenate(zs), np.concatenate(xs))
fn = a * z["frame.number"].values + b
assert np.max(np.abs(fn - np.round(fn))) < 1e-3, "frame numbers are not integers after inversion"
out = pd.DataFrame({"frame_number": np.round(fn).astype(np.int64)})
assert out.frame_number.is_unique
report["columns"]["frame_number"] = {"scale": a, "offset": b, "max_abs_error_on_attack_rows": err,
                                     "min": int(out.frame_number.min()), "max": int(out.frame_number.max())}
out["label"] = z["label"].values

# Join every attack row to its raw record through the restored frame number.
anchors = pd.concat([d.assign(label=k) for k, d in raw.items()], ignore_index=True)
anchors = anchors.merge(out.assign(_i=np.arange(len(out))), left_on=["frame.number", "label"],
                        right_on=["frame_number", "label"], how="left")
assert anchors._i.notna().all(), "an attack packet was not found in the standardized table"
ai = anchors._i.astype(int).values

# ---- frame.time_delta: continuous, many anchors
a, b, err = fit_affine(z["frame.time_delta"].values[ai], anchors["frame.time_delta"].astype(float).values)
td = a * z["frame.time_delta"].values + b
out["frame_time_delta"] = np.round(td, 9).clip(min=0)
report["columns"]["frame_time_delta"] = {"scale": a, "offset": b, "max_abs_error_on_attack_rows": err,
                                         "min_before_clip": float(td.min())}

# ---- data.len: three distinct raw values in the SYN flood capture; missing values were mean-imputed (z == 0)
m = anchors["data.len"].notna().values
a, b, err = fit_affine(z["data.len"].values[ai][m], anchors["data.len"].astype(float).values[m])
dl = a * z["data.len"].values + b
miss = np.abs(z["data.len"].values) < 1e-9
assert np.max(np.abs(dl[~miss] - np.round(dl[~miss]))) < 1e-2
out["data_len"] = nullable_int(dl, miss)
report["columns"]["data_len"] = {"scale": a, "offset": b, "max_abs_error_on_attack_rows": err,
                                 "missing_rows": int(miss.sum())}

# ---- tcp.hdr_len: single raw anchor (20 bytes in the SYN flood); the level spacing gives the 4-byte unit
lv = levels(z["tcp.hdr_len"])
real = lv[np.abs(lv) > 1e-9]                      # the ~0 level is the imputed mean
unit = np.min(np.diff(real))                      # smallest gap between real levels = 4 bytes
z20 = float(np.median(z["tcp.hdr_len"].values[ai][anchors["tcp.hdr_len"].notna().values]))
hl = 20 + (z["tcp.hdr_len"].values - z20) / unit * 4
miss = np.abs(z["tcp.hdr_len"].values) <= 1e-9
assert np.max(np.abs(hl[~miss] - np.round(hl[~miss]))) < 1e-3
out["tcp_hdr_len"] = nullable_int(hl, miss)
report["columns"]["tcp_hdr_len"] = {"anchor": "20 bytes (SYN flood capture)", "unit_bytes": 4,
                                    "levels_recovered": sorted(int(v) for v in pd.Series(out.tcp_hdr_len).dropna().unique()),
                                    "missing_rows": int(miss.sum())}

# ---- tcp.seq: single raw anchor (0 in the SYN flood). The smallest gap between levels is one sequence
# unit; that first estimate is refined by least squares over progressively larger values.
sv = z["tcp.seq"].values
miss = np.abs(sv) < 1e-9                              # mean-imputed rows (no TCP layer)
lv = levels(pd.Series(sv[~miss]))
unit = np.min(np.diff(lv))
z0 = float(np.median(sv[ai][anchors["tcp.seq"].notna().values]))
for limit in (1e4, 1e6, 1e8, 1e10):
    q = (lv - z0) / unit
    m = (q > 0.5) & (q < limit)
    k = np.round(q[m])
    unit = float(np.sum((lv[m] - z0) * k) / np.sum(k * k))
sq = (sv - z0) / unit
frac = np.abs(sq[~miss] - np.round(sq[~miss]))
assert frac.max() < 1e-2, f"tcp.seq not integer after inversion: {frac.max()}"
out["tcp_seq"] = nullable_int(sq, miss)
report["columns"]["tcp_seq"] = {"anchor": "0 (SYN flood capture)", "unit": unit,
                                "max_distance_from_integer": float(frac.max()),
                                "missing_rows": int(miss.sum()), "max": int(np.round(sq[~miss]).max())}

# ---- TCP flags: two levels each -> 0/1 (v1 stored the bit masks 0/2/4/8/16 before scaling)
for f in FLAGS:
    lv = levels(z[f"tcp.flag_{f}"])
    assert len(lv) == 2
    out[f"tcp_flag_{f}"] = (z[f"tcp.flag_{f}"].values > lv.mean()).astype(np.int8)
assert (out.loc[out.label == 4, "tcp_flag_syn"] == 1).all()
report["columns"]["tcp_flags"] = {"note": "binary; verified SYN=1 on every SYN flood row"}


def codes(col):
    """Label-encoded categorical: return integer code per row (0 = lowest level)."""
    lv = levels(z[col])
    return np.searchsorted(lv, z[col].values), len(lv)


# ---- arp.opcode: LabelEncoder over strings sorts 'nan' < 'reply' < 'request'
c, n = codes("arp.opcode")
assert n == 3
op = np.array([None, "reply", "request"], dtype=object)[c]
assert (op[ai] == anchors["arp.opcode"].where(anchors["arp.opcode"].notna(), None).values).all()
out["arp_opcode"] = op
report["columns"]["arp_opcode"] = {"verified_on_attack_rows": True}

# ---- icmp.type: four levels; only type 0 has a raw anchor
c, n = codes("icmp.type")
assert n == 4 and (c[ai][anchors["icmp.type"].notna().values] == 0).all()
out["icmp_type_code"] = nullable_int(c, c == 3)
report["columns"]["icmp_type_code"] = {
    "0": "ICMP type 0, echo reply (verified on raw PING flood capture)",
    "1_and_2": "two other ICMP types seen only in Normal traffic; original type numbers are not recoverable",
    "counts": {str(k): int(v) for k, v in pd.Series(c[c != 3]).value_counts().sort_index().items()}}

# ---- MAC addresses: pseudonymized roles. Known devices come from the raw captures.
for col, new, known in [("arp.src.hw_mac", "arp_src_device", {ATTACKER_MAC: "attacker"}),
                        ("arp.dst.hw_mac", "arp_dst_device",
                         {VICTIM_MAC: "victim", "00:00:00_00:00:00": "unresolved"})]:
    c, n = codes(col)
    name = {n - 1: None}                              # highest code is the string 'nan' (no ARP layer)
    seen = anchors[col].notna().values
    for mac, role in known.items():
        hit = np.unique(c[ai][seen & (anchors[col] == mac).values])
        assert len(hit) == 1, f"{mac} did not map to one code"
        name[int(hit[0])] = role
    k = 0
    for code in range(n - 1):
        if code not in name:
            k += 1
            name[code] = f"other_{k}"
    out[new] = np.array([name[i] for i in range(n)], dtype=object)[c]
    report["columns"][new] = {"levels": {str(v): int((out[new] == v).sum()) for v in name.values() if v}}

# ---- final table, in capture order
out["label_name"] = out.label.map(LABELS)
cols = ["frame_number", "frame_time_delta", "arp_opcode", "arp_src_device", "arp_dst_device", "icmp_type_code",
        "tcp_seq", "tcp_hdr_len", "data_len"] + [f"tcp_flag_{f}" for f in FLAGS] + ["label", "label_name"]
out = out[cols].sort_values("frame_number").reset_index(drop=True)

# Chronological split per class: the first 80% of each class's packets train, the last 20% test.
rank = out.groupby("label").cumcount()
size = out.groupby("label").label.transform("size")
out["split"] = np.where(rank < np.floor(size * 0.8), "train", "test")

os.makedirs(os.path.join(OUT, "raw_attack_captures"), exist_ok=True)
out.to_csv(os.path.join(OUT, "network_attack_traffic.csv"), index=False)
out[out.split == "train"].drop(columns="split").to_csv(os.path.join(OUT, "train.csv"), index=False)
out[out.split == "test"].drop(columns="split").to_csv(os.path.join(OUT, "test.csv"), index=False)
frac = 25000 / len(out)
sample = out.groupby("label", group_keys=False).sample(frac=frac, random_state=42).sort_values("frame_number")
sample.to_csv(os.path.join(OUT, "sample_25k.csv"), index=False)

# Raw attack captures with device names replaced by roles.
MAC_COLS = ["eth.src", "eth.dst", "arp.src.hw_mac", "arp.dst.hw_mac"]
ROLE = {ATTACKER_MAC: "attacker", VICTIM_MAC: "victim", "00:00:00_00:00:00": "unresolved", "Broadcast": "broadcast"}
for k, d in raw.items():
    d = d.copy()
    for col in MAC_COLS:
        unknown = set(d[col].dropna().unique()) - set(ROLE)
        assert not unknown, f"unexpected device in {col}: {unknown}"
        d[col] = d[col].map(ROLE)
    d.insert(0, "label_name", LABELS[k])
    d.insert(0, "label", k)
    d.columns = [c.replace(".", "_") for c in d.columns]
    d.sort_values("frame_number").to_csv(os.path.join(OUT, "raw_attack_captures", f"{RAW_OUT_NAME[k]}.csv"), index=False)

counts = out.label.value_counts().sort_index()
windows = out.groupby("label").frame_number.agg(["min", "max"])
pd.DataFrame({
    "label": counts.index, "label_name": [LABELS[i] for i in counts.index], "count": counts.values,
    "percent": (counts.values / len(out) * 100).round(4),
    "first_frame": windows["min"].values, "last_frame": windows["max"].values,
    "train_rows": out[out.split == "train"].label.value_counts().sort_index().values,
    "test_rows": out[out.split == "test"].label.value_counts().sort_index().values,
}).to_csv(os.path.join(OUT, "label_mapping.csv"), index=False)

feat = [c for c in cols if c not in ("label", "label_name")]
qc = {
    "version": "3.0.0", "n_rows": len(out), "n_feature_columns": len(feat), "n_classes": 5,
    "units": "original Wireshark units; no scaling applied",
    "missing_values_per_column": {c: int(out[c].isna().sum()) for c in feat},
    "duplicate_rows_ignoring_frame_number": int(out.drop(columns=["frame_number", "split"]).duplicated().sum()),
    "frame_number_unique": bool(out.frame_number.is_unique),
    "split": {"rule": "per class, first 80% of packets in capture order -> train, last 20% -> test",
              "train_rows": int((out.split == "train").sum()), "test_rows": int((out.split == "test").sum())},
}
with open(os.path.join(OUT, "dataset_qc_report.json"), "w") as fh:
    json.dump(qc, fh, indent=2)
with open(os.path.join(OUT, "recovery_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report, indent=1))
print(json.dumps(qc, indent=1))
print(out.head(8).to_string())
print(pd.read_csv(os.path.join(OUT, "label_mapping.csv")).to_string())
