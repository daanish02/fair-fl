"""FairWeight's tabular datasets, preprocessed and split exactly as the official code (load_data_utilities.py,
'random' distribution): label-encode categoricals, standardise numericals on the whole dataset, keep the sensitive
attribute as a feature, hold out 20% as the global test set (random_state=42), then carve n clients from the rest
with successive train_test_split(test_size=1/(n-i), random_state=42). CSVs from the authors' release go in
<data.root>/fairweight/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SPECS = {
    "adult": dict(file="adult.csv", label="income", sensitive="sex",
                  categorical=["workclass", "education", "marital.status", "occupation", "relationship", "race",
                               "sex", "native.country"],
                  numerical=["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]),
    "bank": dict(file="bank-full.csv", label="y", sensitive="marital",
                 categorical=["job", "marital", "education", "default", "housing", "loan", "contact", "month",
                              "previous", "poutcome"],
                 numerical=["age", "balance", "day", "duration", "campaign", "pdays"]),
    "default": dict(file="default.csv", label="y", sensitive="SEX",
                    categorical=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
                                 "PAY_5", "PAY_6"],
                    numerical=["AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                               "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]),
    "law": dict(file="law.csv", label="y", sensitive="sex", flip=["y", "sex"],
                categorical=["decile1b", "decile3", "fulltime", "fam_inc", "sex", "race", "tier"],
                numerical=["lsat", "ugpa", "zfygpa", "zgpa"]),
}


def load_fairweight_frame(name: str, root: str | Path):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    spec = SPECS[name]
    path = Path(root) / "fairweight" / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: copy datasets/{spec['file']} from the authors' fairweight.zip")
    data = pd.read_csv(path)
    for col in spec.get("flip", []):
        data[col] = data[col].replace({0: 1, 1: 0})
    for col in spec["categorical"]:
        data[col] = LabelEncoder().fit_transform(data[col])
    data[spec["numerical"]] = StandardScaler().fit_transform(data[spec["numerical"]])
    X = data.drop(spec["label"], axis=1)
    y = LabelEncoder().fit_transform(data[spec["label"]])
    return X, y, spec["sensitive"]


def build_fairweight(name: str, root: str | Path, num_clients: int):
    from sklearn.model_selection import train_test_split

    from fairfl.data.datasets import TensorDataset3
    from fairfl.data.federated import ClientData, FederatedData

    X, y, sens = load_fairweight_frame(name, root)

    def tds(Xd, yd):
        return TensorDataset3(torch.tensor(Xd.values, dtype=torch.float32), torch.tensor(np.asarray(yd)).long(),
                              torch.tensor(Xd[sens].values).long(), 2, 2)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clients = []
    for i in range(num_clients):
        if i == num_clients - 1:
            Xc, yc = X_temp, y_temp
        else:
            X_temp, Xc, y_temp, yc = train_test_split(X_temp, y_temp, test_size=1 / (num_clients - i), random_state=42)
        train = tds(Xc, yc)
        clients.append(ClientData(i, train, train.subset(np.array([], dtype=np.int64))))
    return FederatedData(clients, (X.shape[1],), 2, 2, tds(X_test, y_test))
