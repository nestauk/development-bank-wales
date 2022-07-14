import sklearn
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from development_bank_wales.pipeline.machine_learning import evaluation, plotting

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
        print("\nBaseline\n************")
        evaluation.print_metrics(test_labels, baseline)

        print("\nTraining Set\n************")
        evaluation.print_metrics(train_labels, y_pred_train)

        print("\nTest Set\n************")
        evaluation.print_metrics(test_labels, y_pred_test)

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
    # features_df['proba {}'.format(label_name)] = probabilities[:, 1]

    return probabilities[:, 1]
