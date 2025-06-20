import csv
import sys

# Get filenames from command line arguments, or use defaults
input_filename = sys.argv[1] if len(sys.argv) > 1 else 'input.csv'
base_filename = input_filename.split('.')[0]
output_filename = f'{base_filename}_sorted.csv'

with open(input_filename, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = sorted(reader, key=lambda x: int(x[0].split('.')[0]))

with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)