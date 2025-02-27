# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import joblib

def load_data():
    # Replace with actual data loading logic
    df = pd.read_csv('merged_churn.csv')
    return df

def preprocess_data(df):
    """
    Preprocess the data by standardizing the continuous columns.
    """
    # Define which columns are continuous (not binary)
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]
    
    # Ensure only numeric columns are processed
    continuous_columns = [col for col in df.columns if col not in binary_columns and df[col].dtype in ['float64', 'int64']]

    # Standardize continuous columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[continuous_columns]), columns=continuous_columns)
    
    return df_scaled


def perform_pca(df_scaled):
    pca = PCA()
    df_pca = pca.fit_transform(df_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_explained_variance)

    return df_pca, explained_variance_ratio, cumulative_explained_variance, pca

def plot_pca_explained_variance(explained_variance_ratio, cumulative_explained_variance):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='skyblue')
    plt.title('Explained Variance by Principal Components', fontsize=14)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', color='red', linestyle='-')
    plt.title('Cumulative Explained Variance by Principal Components', fontsize=14)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.xticks(range(1, len(cumulative_explained_variance) + 1))
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.show()

def perform_kmeans_clustering(df_scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, init='random', n_init=3, random_state=109)
    model.fit(df_scaled)

    wss = []
    for i in range(1, 11):
        fitx = KMeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(df_scaled)
        wss.append(fitx.inertia_)

    plt.figure(figsize=(11, 8.5))
    plt.plot(range(1, 11), wss, 'bx-')
    plt.xlabel('Number of clusters $k$')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal $k$')
    plt.show()

    return model

def hierarchical_clustering(df_scaled, num_clusters=3):
    linkage_matrix = linkage(df_scaled, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.grid()
    plt.show()

    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    return cluster_labels

def train_random_forest(X_train_scaled, Y_train, X_test_scaled, Y_test):
    param_grid_rf = {
        'n_estimators': [50],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [False]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1)
    grid_search_rf.fit(X_train_scaled, Y_train)

    print(f"Best parameters from Grid Search: {grid_search_rf.best_params_}")

    best_rf = grid_search_rf.best_estimator_
    print(f"Model type: {type(best_rf)}")
    Y_pred_rf = best_rf.predict(X_test_scaled)
    y_probs_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy: {accuracy_score(Y_test, Y_pred_rf)}")
    print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred_rf)}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred_rf)}")

    cm_rf = confusion_matrix(Y_test, Y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test, y_probs_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='blue', label=f'ROC curve (area = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
    plt.legend(loc="lower right")
    plt.show()

    return best_rf
def save_model(model, filename="model.joblib"):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
