import os

label_dir = "C:/Users/Microsoft/Downloads/data-20250209T153416Z-001/data/labels/train"
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                values = line.strip().split()
                if len(values) != 26:  # Expected 26 columns per row
                    print(f"❌ Incorrect label format in {filename}, line {line_num}: {line}")
