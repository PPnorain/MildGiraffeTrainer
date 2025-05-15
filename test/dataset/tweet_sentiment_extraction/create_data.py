from datasets import DatasetDict, load_dataset
import csv
import json

def main():
    label2id = {"positive": 2, "neutral": 1, "negative": 0}

    for split in ["train", "test"]:
        input_file = csv.DictReader(open(f"raw_data/{split}_csv"))

        with open(f'{split}.jsonl', 'w') as fOut:
            for row in input_file:
                fOut.write(json.dumps({'id': row['textID'], 'text': row['text'], 'label': label2id[row['sentiment']], 'label_text': row['sentiment']})+"\n")
               

    """
    train_dset = load_dataset("csv", data_files="raw_data/train_csv", split="train")
    train_dset = train_dset.remove_columns(["selected_text"])
    test_dset = load_dataset("csv", data_files="raw_data/train_csv", split="train")
    raw_dset = DatasetDict()
    raw_dset["train"] = train_dset
    raw_dset["test"] = test_dset
    for split, dset in raw_dset.items():
        dset = dset.rename_column("sentiment", "label_text")
        dset = dset.map(lambda x: {"label": label2id[x["label_text"]]}, num_proc=8)
        dset.to_json(f"{split}.jsonl")
    """

if __name__ == "__main__":
    main()