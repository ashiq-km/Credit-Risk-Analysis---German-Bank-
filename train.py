import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns


from src.draw_plot import Threshold_Max





def train():

    # ---- MLFlow Experiment Setup ----

    mlflow.set_experiment("German Credit Risk")

    # --- MLFlow Run ---

    with mlflow.start_run():

        df = pd.read_csv(r"data/gd.csv")

        df['Credit_risk'] = df['Credit_risk'].map({2:1, 1:0})

        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




        # Defining the order of priority that have to be labelled in Ordinal Encoding

        ordinal_categories = [
            ["A61", "A62", "A63", "A64", "A65"], 
            ["A71", "A72", "A73", "A74", "A75"], 
            ["A124", "A123", "A122", "A121"], 
            ["A171", "A172", "A173", "A174"]
        ]

        #  OHE --------- Columns while pre-processing  ----------------

        ohe_columns = ["Account_status", "Credit_history", "loan_Purpose", "Sex", 
                    "Marital_status", "co-debtors", "Other_loans", "Housing", 
                    "Dependents", "Telephone", "Foreign_worker"]


        # -------------- Scaling for RobustScaler ------------------


        rob_scaling = ["Duration_months", "Credit_amount", "Age"]

        ord_enc = ["Savings/Bonds_AC", "Present_Employment_since", "Assets/Physical_property", 
                "Job_status"]



        # ------------ (❁´◡`❁) ----------------- Processing ----------------

        ct = ColumnTransformer(
            transformers = [
                ("cat", OneHotEncoder(drop = "if_binary", sparse_output = False, handle_unknown="ignore"), ohe_columns), 
                ("scale", RobustScaler(), rob_scaling),
                ("enc", OrdinalEncoder(categories=ordinal_categories), ord_enc)
            ], 
            remainder="passthrough"
        )



        # ------------ (❁´◡`❁) ----------------- Basic Pipeline Modelling ----------------


        base_pipeline = Pipeline(
            [
                ("preprocess", ct), 
                ("smote", BorderlineSMOTE(
                    sampling_strategy=0.8, 
                    random_state=42
                )), 
                ("model", XGBClassifier(
                    tree_method = "hist", 
                ))
            ]
        )


        # ------------ (￣y▽￣)╭ Ohohoho..... ----------------- Hyper Paramater Tuning --- Parameters Defined ----------------



        new_param_xgb = {
            "model__n_estimators": randint(100, 1000), 
            "model__learning_rate": uniform(0.005, 0.080), 
            "model__max_depth": randint(3, 12), 
            "model__min_child_weight": randint(1, 10), 
            "model__gamma": uniform(0, 0.5), 
            "model__subsample": uniform(0.6, 0.4), 
            "model__colsample_bytree": uniform(0.7, 0.3), 
            "model__scale_pos_weight": uniform(2, 1), 
            "smote__sampling_strategy": uniform(0.7, 0.3)
        }




        random_xgb = RandomizedSearchCV(
            estimator = base_pipeline, 
            param_distributions  = new_param_xgb, 
            scoring = "recall", 
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
            n_jobs=-1, 
            random_state=42, 
            verbose=3, 
            n_iter = 300, 
        )

        print("Startine Training.....\n")


        # --- 👽RandomizedSearchCV Training👽 ---

        random_xgb.fit(X_train, y_train)





        # --- Metric to be logged in MLFlow ---

        best_model = random_xgb.best_estimator_
        best_params = random_xgb.best_params_
        best_score = random_xgb.best_score_


        # Calcultate the test set Metric:

        y_pred = random_xgb.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        report_text = classification_report(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)


        # --- Confusion Matrix Plot ---

        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt = "d", cmap = "Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()



        # --- precision-recall curve ---

        y_scores = random_xgb.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

        plt.figure(figsize= (7, 5))
        plt.plot(recall, precision, linewidth = 2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)

        plt.savefig('pr_curve.png')
        plt.close()


        

        print(f"Best Params: {best_params}")
        print(f"Test Accuracy : {test_accuracy}")
        print(classification_report(y_test, y_pred))


        # ---- MLFLOW Metrics | Paramaters | Text Log ---

        mlflow.log_params(best_params)

        mlflow.log_metric("Test Accuracy Before selecting Threshold", test_accuracy)
        mlflow.log_metric("Best Score before selecting Threshold", best_score)
        mlflow.log_text(report_text, "Classification_report_b4 threshold selection.txt")




        # --- MLFLOW Artifacts ---

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact('pr_curve.png')


        # --- MLFlow JSON Dict ---

        mlflow.log_dict(best_params, "Model params B4 Threshold Selection.json")



        # --- MLFlow logged Models ---

        mlflow.sklearn.log_model(best_model, "Initial Model B4 Threshold Selection")

        new_exp = Threshold_Max(random_xgb, X_train, X_test, y_train, y_test)
        new_exp.recall_plot()



        model_path = "models/xgb_pipeline.joblib"


        joblib.dump(random_xgb.best_estimator_, model_path)


        print(f"Model Trained and Saved To {model_path}")



        



if __name__ == "__main__":
    train()

