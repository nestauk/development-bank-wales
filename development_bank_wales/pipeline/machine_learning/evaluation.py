import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import matplotlib.pyplot as plt
import seaborn as sns


def print_metrics(labels, predictions):

    acc = np.round(accuracy_score(labels, predictions) * 100, 2)
    f1 = np.round(f1_score(labels, predictions) * 100, 2)
    precision = np.round(precision_score(labels, predictions) * 100, 2)
    recall = np.round(recall_score(labels, predictions) * 100, 2)

    print("Accuracy:\t{}%".format(acc))
    print("F1 score:\t{}%".format(f1))
    print("Recall:\t\t{}%".format(precision))
    print("Precision:\t{}%".format(recall))


def decision_tree_plotting():

    """Not tested"""

    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot

    # Pull out one tree from the forest
    tree = model.estimators_[5]
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot

    # Pull out one tree from the forest
    tree = model.estimators_[5]
    # Export the image to a dot file
    export_graphviz(
        tree, out_file="tree.dot", feature_names=feature_list, rounded=True, precision=1
    )
    # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file("tree.dot")
    # Write graph to a png file
    graph.write_png("tree.png")

    # Limit depth of tree to 3 levels
    rf_small = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf_small.fit(train_features, train_labels)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(
        tree_small,
        out_file="small_tree.dot",
        feature_names=feature_list,
        rounded=True,
        precision=1,
    )
    (graph,) = pydot.graph_from_dot_file("small_tree.dot")
    graph.write_png("small_tree.png")

    # Get numerical feature importances
    importances = list(
        model.feature_importances_
    )  # List of tuples with variable and importance
    feature_importances = [
        (feature, round(importance, 2))
        for feature, importance in zip(feature_list, importances)
    ]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [
        print("Variable: {:20} Importance: {}".format(*pair))
        for pair in feature_importances
    ]


def get_baseline(n_samples, true_rate):

    baseline = np.zeros((n_samples))
    n_true = n_samples * true_rate
    n_trues_indices = np.random.choice(
        range(0, n_samples), round(n_true), replace=False
    )
    baseline[n_trues_indices] = 1.0
    return baseline
