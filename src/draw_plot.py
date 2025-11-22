import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, classification_report
import mlflow, mlflow.sklearn
import numpy as np


class ThresholdedModel(mlflow.pyfunc.PythonModel):
     
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        proba = self.model.predict_proba(model_input)[:, 1]
        return (proba>=self.threshold).astype('int')


class Threshold_Max:

    def __init__(self, search, X_train, X_test, y_train, y_test):

        self.search = search
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train


    def recall_plot(self):

        y_pred = self.search.predict(self.X_test)
        train_pred = self.search.predict(self.X_train)

        y_scores = self.search.predict_proba(self.X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(self.y_test, y_scores)


        # --- Precision-Recall Threshold plot ---


        plt.figure(figsize = (7, 5))
        plt.plot(thresholds, precision[1:], label = "Precision")
        plt.plot(thresholds, recall[1:], label = "Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Precision/ Recall")
        plt.title("Precision-Recall Threshold Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("precision-recall_threshold.png")
        
        plt.close()



        # --- Best threshold from precision-recall threshold curve
        try:

            threshold_index  = np.where(recall[1:]>0.90)[0][-1]

            best_threshold = thresholds[threshold_index]

        except:
                best_threshold = None
        ## I want precision-recall threshold curve, but automation as well, 
        ## i am wishing for more than 85% recall, 
        ## We can look for automatic f1_score, like which gives the best, at what threshold we can see


        ## At the same time, we can also see what threshold gives what recall as well, 
        # Also we have to log all of these, so, let's start with logging train and test scores first.

        train_best_score = self.search.best_score_
        train_recall_score = recall_score(self.y_train,train_pred)
        test_recall_score = recall_score(self.y_test, y_pred)


        # --- f1-Score Plot ---

        f1 = []
        for i in thresholds:
            f1_pred = (y_scores>=i).astype('int') # This creates a boolean array
            f1_sc = f1_score(self.y_test, f1_pred)
            f1.append(f1_sc)



        # --- F1 Score - Threshold Plot

        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, f1)
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 - Threshold Curve")
        plt.grid(True)
        plt.savefig("f1_Threshold Curve.png")
        plt.close()

        f1_best_index = np.argmax(f1)
        f1_best_score = f1[f1_best_index]
        f1_best_threshold = thresholds[f1_best_index]


        # --- mlflow metrics---

        mlflow.log_metrics(
            {
            "Model Best Score": train_best_score, 
            "Recall - Train": train_recall_score, 
            "Recall - Test": test_recall_score, 
            "Best Threshold after selection ": best_threshold, 
            "F1 Best Threshold": f1_best_threshold, 
            "F1 Best Score": f1_best_score
        }
        )



        # --- mlflow artifacts ---

        mlflow.log_artifact("precision-recall_threshold.png")
        mlflow.log_artifact("f1_Threshold Curve.png")



        # --- ml model ---

        # mlflow.sklearn.log_model(self.search.best_estimator_, "model")

        final_prediction = (y_scores>=best_threshold).astype('int')


        final_report = classification_report(self.y_test, final_prediction)


        wrapped = self.get_mlflow_ready_model(best_threshold)

        # -- mlflow pyfunc.PythonModel model ---

        mlflow.pyfunc.log_model(name="Thresholded Model", python_model = wrapped)



        mlflow.log_text(final_report, "Classification_Final.txt")


    def get_mlflow_ready_model(self, threshold):
            return ThresholdedModel(self.search.best_estimator_, threshold)



