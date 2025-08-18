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

def main(csv_path, toml_path):
    data = csv_to_dicts(csv_path)
    write_toml(data, toml_path)
    print(f"Converted {csv_path} → {toml_path}")

if __name__ == "__main__":
    main("/Users/wangyichi/Documents/Projects/math_tutor/student_data.csv", "/Users/wangyichi/Documents/Projects/math_tutor/student_data.toml")