import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import (save_model, load_data, preprocess_data, perform_pca,
                            plot_pca_explained_variance, perform_kmeans_clustering,
                            hierarchical_clustering, train_random_forest)
from sklearn.model_selection import train_test_split

def main(args):
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run():
        # Step 1: Load data
        df = load_data()
        mlflow.log_param("data_loaded", "True")

        # Step 2: Preprocess data (standardization)
        df_scaled = preprocess_data(df)
        mlflow.log_param("data_prepared", True)

        if args.pca:
            # Step 3: Perform PCA
            df_pca, explained_variance_ratio, cumulative_explained_variance, pca = perform_pca(df_scaled)
            # Step 4: Plot PCA explained variance
            plot_pca_explained_variance(explained_variance_ratio, cumulative_explained_variance)
            mlflow.log_param("PCA_components", pca.n_components_)


        if args.kmeans:
            # Step 5: Perform KMeans clustering
            model_kmeans = perform_kmeans_clustering(df_scaled)
            mlflow.log_param("KMeans_clusters", model_kmeans.n_clusters)

        if args.hierarchical:
            # Step 6: Perform Hierarchical Clustering
            cluster_labels = hierarchical_clustering(df_scaled)
            df['Cluster'] = cluster_labels  # Add the cluster labels to the dataframe
            mlflow.log_param("Hierarchical_clustering_done", True)

        if args.train:
            # Step 7: Train Random Forest
            X = df_scaled
            Y = df['Churn']  # Ensure that 'Churn' is a valid column

            # Split the data into training and testing sets
            X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            model_rf = train_random_forest(X_train_scaled, Y_train, X_test_scaled, Y_test)

            # Log model and parameters
            mlflow.log_param("RandomForest_n_estimators", model_rf.n_estimators)
            mlflow.log_param("RandomForest_max_depth", model_rf.max_depth)
            mlflow.sklearn.log_model(model_rf, "random_forest_model")
            save_model(model_rf)

        if args.evaluate:
            print("Evaluation is not yet implemented.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full machine learning pipeline")
    parser.add_argument('--load', action='store_true', help="Load the data")
    parser.add_argument('--prepare', action='store_true', help="Prepare the data (standardization)")
    parser.add_argument('--pca', action='store_true', help="Perform PCA and plot explained variance")
    parser.add_argument('--kmeans', action='store_true', help="Perform KMeans clustering")
    parser.add_argument('--hierarchical', action='store_true', help="Perform Hierarchical Clustering")
    parser.add_argument('--train', action='store_true', help="Train the Random Forest model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained model")

    args = parser.parse_args()
    main(args)
