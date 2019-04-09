from sklearn import svm
from sklearn import metrics
import numpy as np

def get_svm_metrics(model,train_data,train_labels,test_data,test_labels):
  #training the model
  model.fit(train_data,train_labels)

  #making a prediction from the trained model
  model_prediction = model.predict(test_data)

  #getting metrics
  acc = metrics.accuracy_score(test_labels, model_prediction)
  cm = metrics.confusion_matrix(test_labels,model_prediction)
  recall = metrics.recall_score(test_labels,model_prediction)
  precision = metrics.precision_score(test_labels,model_prediction)
  report = metrics.classification_report(test_labels,model_prediction)

  print("The results of the",model, "model are:")
  print("accuracy:", acc)
  print("recall:", recall)
  print("precision", precision)
  print("The confusion matrix:", cm)
  print("Lastly the Classification Report:", report)

  # vim: set ts=2 sw=2 expandtab:

