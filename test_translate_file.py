#!/usr/bin/env python3

from preproc import translate_file

if __name__ == '__main__':
  input_file = open('Data/new_data.csv', 'r')
  output_file = open('/tmp/test_data.csv', 'w')
  translate_file(input_file, output_file)
