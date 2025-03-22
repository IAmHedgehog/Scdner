import sys
import csv


def parsecsv(csv_file):
    best_row = None
    head_row = None
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for idx, row in enumerate(reader):
            if idx == 0:
                head_row = row
            elif not best_row or best_row[2] < row[2]:
                best_row = row
    print('Best epoch: ', best_row[6])
    print(dict(zip(head_row[:6], [round(float(num), 2) for num in best_row[:6]])))


if __name__ == '__main__':
    args = sys.argv[1:]
    file_name = args[0]
    parsecsv('./data/log/%s/eval_valid.csv' % file_name)
