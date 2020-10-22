import csv
import os
from io import TextIOWrapper
from zipfile import ZipFile


def main():

    for raw_data_path in ["test.csv.zip", "train.csv.zip"]:

        with ZipFile(raw_data_path) as zf:
            csv_file_path = os.path.basename(raw_data_path).split(".zip")[0]
            output_content_path = csv_file_path.split(".csv")[0] + "_content.txt"
            output_label_path = csv_file_path.split(".csv")[0] + "_label.txt"

            with zf.open(os.path.basename(csv_file_path), "r") as infile:
                reader = csv.reader(TextIOWrapper(infile, "utf-8"))
                next(reader)
                with open(output_content_path, "w") as content_file:
                    with open(output_label_path, "w") as label_file:

                        for index, title, content in reader:
                            content_file.write(content + "\n")
                            label_file.write(index + "\n")


if __name__ == "__main__":
    main()
