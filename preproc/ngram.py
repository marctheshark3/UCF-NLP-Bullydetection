from translate import translate_file
import pandas as pd

file_location = "/Users/marctheshark/Documents/Github/NLP/Tokenization/data.txt"
output = "/Users/marctheshark/Documents/Github/NLP/Tokenization/output.txt"

data = translate_file(file_location, output)

