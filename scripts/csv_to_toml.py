import csv
import sys
import toml

def csv_to_dicts(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_toml(data, toml_path):
    with open(toml_path, 'w', encoding='utf-8') as f:
        toml.dump({"rows": data}, f)

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    csv_path, toml_path = sys.argv[1], sys.argv[2]
    data = csv_to_dicts(csv_path)
    write_toml(data, toml_path)
    print(f"Converted {csv_path} → {toml_path}")

if __name__ == "__main__":
    main()