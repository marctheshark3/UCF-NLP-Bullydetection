class Metrics(object):
  """
  Namespace containing symbolic equivalents for all the dictionary keys used
  to store model evaluation metrics.

  :cvar FPR: Symbol for the "False Positive Rate" metric dictionary key.
  :vartype FPR: str
  :cvar FNR: Symbol for the "False Negative Rate" metric dictionary key.
  :vartype FNR: str
  :cvar TNR: Symbol for the "True Negative Rate" metric dictionary key.
  :vartype TNR: str
  :cvar ACCURACY: Symbol for the "Accuracy" metric dictionary key.
  :vartype ACCURACY: str
  :cvar PRECISION: Symbol for the "Precision" metric dictionary key.
  :vartype PRECISION: str
  :cvar RECALL: Symbol for the "Recall", also known as "True Positive Rate",
  metric dictionary key.
  :vartype RECALL: str
  """
  FPR = "FPR"
  FNR = "FNR"
  TNR = "TNR"
  ACCURACY = "Accuracy"
  PRECISION = "Precision"
  RECALL ="Recall" # a.k.a. True Positive Rate

# vim: set ts=2 sw=2 expandtab:
