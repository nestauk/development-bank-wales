# File: development_bankd_wales/pipeline/predictive_model/training.py
"""
Train and evaluate different predictive models.
"""
# ----------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from development_bank_wales.pipeline.predictive_model import evaluation, plotting

# ----------------------------------------------------------------------------------

model_dict = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
    "Support Vector Classifier": svm.SVC(probability=True, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
}


def train_and_evaluate_model(
    features, labels, model_name, label_name, feature_list, verbose=True
):
    """Train and evaluate a predictive model.
    Information about the training and evaluation process is printed by default.
    The predicted probabilties for all features are returned.

    Args:
        features (np.array): Feature set used for training model.
        labels (np.array): Ground truth labels.
        model_name (str): Model to be used.
        label_name (str): Name of the label ("what are we predicting").
        feature_list (list): List of feature names.
        verbose (bool, optional): Whether to print model, data and evaluation information. Defaults to True.

    Returns:
        probabilities (np.array): Probabilities (for binary classification task) for all features.
    """

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )

    if verbose:

        print("Predicting {} with {}".format(label_name, model_name))
        print("***************************************\n")
        print("# Training Samples:", train_features.shape[0])
        print("# Testing Samples:", test_features.shape[0])
        print("# Features:", train_features.shape[1])

    model = model_dict[model_name]
    model.fit(train_features, train_labels)

    y_pred_test = model.predict(test_features)
    y_pred_train = model.predict(train_features)
    accuracy_score(test_labels, y_pred_test)

    # Baseline
    true_rate = labels.sum() / labels.shape[0]
    baseline = evaluation.get_baseline(test_labels.shape[0], true_rate)

    if verbose:
        print("\nBaseline: \n************")
        evaluation.print_metrics(test_labels, baseline)

        print("\nTraining Set:\n************\n")
        evaluation.print_cross_validation(
            model_dict[model_name], train_features, train_labels, cv=5
        )
        # evaluation.print_metrics(train_labels, y_pred_train)

        print("\nTest Set:\n************\n")
        evaluation.print_cross_validation(
            model_dict[model_name], test_features, test_labels, cv=5
        )
        # evaluation.print_metrics(test_labels, y_pred_test)

    plotting.plot_confusion_matrix(
        test_labels,
        y_pred_test,
        label_set=["Not upgradable", "Upgradable"],
        title="Predicting {} with {}".format(label_name, model_name),
        plot_type="plt",
    )

    plotting.get_most_important_coefficients(
        model,
        feature_list,
        "Feature Importance for predicting {}".format(label_name),
        train_features,
        top_features=12,
    )

    probabilities = model.predict_proba(features)

    return probabilities[:, 1]
