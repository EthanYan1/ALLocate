from pathlib import Path
import yaml

yaml_dir = Path("data/fold_yamls")
yaml_dir.mkdir(parents=True, exist_ok=True)

n_folds = 5

names = ['Artifact', 'Atypical', 'Typical']

for val_fold in range(1, n_folds + 1):
    train_list = []
    for fold in range(1, n_folds + 1):
        if fold != val_fold:
            train_list.append(f"../folds/fold{fold}/images")

    data = {
        "train": train_list,
        "val": f"../folds/fold{val_fold}/images",
        "test": "../test/images",
        "names": names
    }

    out_file = yaml_dir / f"data_fold{val_fold}.yaml"
    with open(out_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Created {out_file}")