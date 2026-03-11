from model_generators.classification.train_classifier import train_classification_model
from model_generators.clustering.train_cluster import train_clustering_model
from model_generators.regression.train_regression import train_regression_model


if __name__ == "__main__":
    train_regression_model(force=True)
    train_classification_model(force=True)
    train_clustering_model(force=True)
    print("All models trained successfully.")
