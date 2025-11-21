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




def train():

    # ---- MLFlow Setup ----

    mlflow.set_experiment("German Credit Risk")

    with mlflow.start_run():

        df = pd.read_csv(r"data/gd.csv")

        df['Credit_risk'] = df['Credit_risk'].map({2:1, 1:0})

        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


        ordinal_categories = [
            ["A61", "A62", "A63", "A64", "A65"], 
            ["A71", "A72", "A73", "A74", "A75"], 
            ["A124", "A123", "A122", "A121"], 
            ["A171", "A172", "A173", "A174"]
        ]


        ohe_columns = ["Account_status", "Credit_history", "loan_Purpose", "Sex", 
                    "Marital_status", "co-debtors", "Other_loans", "Housing", 
                    "Dependents", "Telephone", "Foreign_worker"]


        rob_scaling = ["Duration_months", "Credit_amount", "Age"]

        ord_enc = ["Savings/Bonds_AC", "Present_Employment_since", "Assets/Physical_property", 
                "Job_status"]


        ct = ColumnTransformer(
            transformers = [
                ("cat", OneHotEncoder(drop = "if_binary", sparse_output = False, handle_unknown="ignore"), ohe_columns), 
                ("scale", RobustScaler(), rob_scaling),
                ("enc", OrdinalEncoder(categories=ordinal_categories), ord_enc)
            ], 
            remainder="passthrough"
        )


        base_pipeline = Pipeline(
            [
                ("preprocess", ct), 
                ("smote", BorderlineSMOTE(
                    sampling_strategy=0.8, 
                    random_state=42
                )), 
                ("model", XGBClassifier(
                    tree_method = "hist"
                ))
            ]
        )



        new_param_xgb = {
            "model__n_estimators": randint(100, 1000), 
            "model__learning_rate": uniform(0.01, 0.1), 
            "model__max_depth": randint(7, 10), 
            "model__min_child_weight": randint(1, 10), 
            "model__gamma": uniform(0, 0.5), 
            "model__subsample": uniform(0.6, 0.4), 
            "model__colsample_bytree": uniform(0.7, 0.3), 
            "model__scale_pos_weight": uniform(2, 1)
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

        random_xgb.fit(X_train, y_train)


        # --- Log Metric to MLFlow ---

        best_model = random_xgb.best_estimator_
        best_params = random_xgb.best_params_
        best_score = random_xgb.best_score_


        # Calcultate the test set Metric:

        y_pred = random_xgb.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        report_text = classification_report(y_test, y_pred)

        

        print(f"Best Params: {best_params}")
        print(f"Test Accuracy: {test_accuracy}")
        print(classification_report(y_test, y_pred))


        mlflow.log_params(best_params)

        mlflow.log_metric("Test Accurcacy", test_accuracy)
        mlflow.log_metric("CV_Accuracy", best_score)
        mlflow.log_text(report_text, "Classification_report.txt")






        model_path = "models/xgb_pipeline.joblib"


        joblib.dump(random_xgb.best_estimator_, model_path)


        print(f"Model Trained and Saved To {model_path}")



if __name__ == "__main__":
    train()

