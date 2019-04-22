from sklearn import svm
from sklearn import metrics
import numpy as np

def get_svm_metrics(model,train_data,train_labels,test_data,test_labels):

  #training the given model
  model.fit(train_data,train_labels)

  #making a prediction from the trained model
  model_prediction = model.predict(test_data)

  #getting metrics

  #need to verify that the ACC metrics works
  #rather look up multiclass metrics like reports
  #potentially look at seperating by false/true postive/negative
  #acc = metrics.accuracy_score(test_labels, model_prediction)

  report = metrics.classification_report(test_labels,model_prediction)
  print("The results of the",model, "model are:")
  print("")
  print("")
  print("The Classification Report:", report)

  prec= report[0]
  recall = report[1]
  f1 = report[2]
  support = report[3]

  print("the precision is:", prec, "and recall", recall)


  return report

  # vim: set ts=2 sw=2 expandtab:

