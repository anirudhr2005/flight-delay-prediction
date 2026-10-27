"""The five course models, each as a Pipeline with the encoding that suits it.

- Logistic Regression, SVM: one-hot airports/airline + scaled numerics
- Decision Tree, Random Forest, Naive Bayes: cross-fitted target encoding of
  airports/airline (one number per category) + numerics
- class_weight="balanced" (where supported) counters the ~3.6:1 imbalance.

Encoders live inside the pipeline, so nothing is fitted on validation or
test rows.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL, NUMERIC, SEED

# Display name -> file name in models/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "SVM (RBF)": "svm",
    "Naive Bayes": "naive_bayes",
}


def _one_hot_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def _target_enc_prep(scale=False):
    return ColumnTransformer([
        ("cat", TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=SEED)), CATEGORICAL),
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("prep", _one_hot_prep()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", _target_enc_prep()),
            ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                           class_weight="balanced", random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", _target_enc_prep()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                           max_features="sqrt", class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED)),
        ]),
        # SVC has no native probabilities; Platt (sigmoid) calibration on
        # internal CV provides them for ROC-AUC and threshold tuning.
        "SVM (RBF)": Pipeline([
            ("prep", _one_hot_prep()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    random_state=SEED),
                method="sigmoid", ensemble=False)),
        ]),
        "Naive Bayes": Pipeline([
            ("prep", _target_enc_prep(scale=True)),
            ("clf", GaussianNB()),
        ]),
    }
