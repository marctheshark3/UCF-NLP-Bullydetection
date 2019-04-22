import csv
import sys
import argparse
import math
import pprint

DEFAULT_SAMPLE_COUNT = 9341

def read_results(results_file):
  processed_results = {}
  results_reader = csv.DictReader(results_file)
  for a_result_row in results_reader:
    proc_name = '{:s}({:s})'.format(
      a_result_row['classifier'],
      a_result_row['ngram']
    )
    proc_result_row = processed_results.get(proc_name, {})
    new_val = proc_result_row.get('recall', [0, 0])
    new_val[0] += float(a_result_row['recall'])
    new_val[1] += 1
    proc_result_row['recall'] = new_val
    processed_results[proc_name] = proc_result_row
  for proc_row in processed_results.values():
    proc_row['recall'] = proc_row['recall'][0] / proc_row['recall'][1] if proc_row['recall'][1] > 0 else 0.0
  return processed_results

def derive_confidence_interval(processed_results, sample_count):
  csv_result = []
  for proc_row_label, proc_row in processed_results.items():
    recall = proc_row['recall']
    variance = math.sqrt((recall * (1. - recall)) / float(sample_count))
    interval = (
      recall - (1.96 * variance),
      recall + (1.96 * variance)
    )
    csv_result.append({
      'name': proc_row_label,
      'recall_low': interval[0],
      'recall_high': interval[1],
      'recall': recall
    })
  return csv_result

def write_interval_result(csv_result):
  csv_writer = csv.DictWriter(sys.stdout, fieldnames=['name', 'recall_low', 'recall_high', 'recall'])
  csv_writer.writeheader()
  for a_csv_row in csv_result:
    csv_writer.writerow(a_csv_row)

parser = argparse.ArgumentParser(description='Statistical Significance Calculation')
parser.add_argument('input_file', nargs=1, help='Results CSV input file')
parser.add_argument('-s', '--sample-count', nargs=1, type=int, default=DEFAULT_SAMPLE_COUNT, help='Use as sample count instead of default {}'.format(DEFAULT_SAMPLE_COUNT))
options = parser.parse_args(sys.argv[1:])
if options.input_file == '-':
  results_file = sys.stdin
else:
  results_file = open(options.input_file[0], 'r')
processed_results = read_results(results_file)
csv_result = derive_confidence_interval(processed_results, options.sample_count)
write_interval_result(csv_result)

if results_file is not sys.stdin:
  results_file.close()

# vim: set ts=2 sw=2 expandtab:
