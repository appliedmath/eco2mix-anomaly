"""The trained model as an MLflow pyfunc, so it can go into the Model Registry.

The saved model is small: for each detector the median and MAD of its
normal error, plus K. predict() turns (actual, expected) into a robust
z-score and an alert flag, exactly like train.make_detection().

    model = mlflow.pyfunc.load_model("models:/grid-anomaly-detector@champion")
    model.predict(df[["actual", "expected"]], params={"detector": "mstl"})
"""
import joblib
import mlflow.pyfunc
import pandas as pd


class AnomalyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        detector = (params or {}).get("detector", "mstl")
        ref, k = self.model["refs"][detector], self.model["k"]
        gap = model_input["actual"] - model_input["expected"]
        z = (gap - ref["median"]) / (1.4826 * ref["mad"])
        return pd.DataFrame({"z_score": z, "is_anomaly": z.abs() > k})
