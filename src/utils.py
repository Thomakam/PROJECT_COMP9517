import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model Evaluation
def evaluate_performance(true_label, pred_label, model_name=None):
	"""
	Evaluates and print model performance based on accuracy, precision, recall, and F1 score.
	
	Arguments:
	- true_label: list or np.ndarray, true labels
	- pred_label: list or np.ndarray, predicted labels
	- model_name: Optional; a string for model description (default is None)
	
	"""
	# error handling
	if type(true_label) != list and type(true_label) != np.ndarray:
		raise TypeError("true_label should be a list or a numpy array.")
	if type(pred_label) != list and type(pred_label) != np.ndarray:
		raise TypeError("pred_label should be a list or a numpy array.")
	
	if len(true_label) != len(pred_label):
		raise ValueError("true_label and pred_label must have the same length.")
	
	if len(true_label) == 0 or len(pred_label) == 0 :
		raise ValueError("true_label and pred_label cannot be empty.")
	
	# Convert inputs to numpy arrays for easy processing
	true_label = np.array(true_label)
	pred_label = np.array(pred_label)

	# Calculate metrics
	accuracy = accuracy_score(true_label, pred_label)
	precision = precision_score(true_label, pred_label, average="weighted", zero_division=0)
	recall = recall_score(true_label, pred_label, average="weighted", zero_division=0)
	f1 = f1_score(true_label, pred_label, average="weighted", zero_division=0)

	# Print results
	if model_name is not None:
		print(f"Model: {model_name}")
	else:
		print(f"Model: default")
		
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print()
