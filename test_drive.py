from n_grams import get_n_grams
from Cross_validation import Cross_validation

## Example of usage :
## 	get n-grams
##  split into validations
vectors, labels = get_n_grams(3, "test_data.csv")

crossval = Cross_validation(vectors, labels, 5)
train,train_labels,test,test_labels = crossval.get_sets(0)