import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from joblib import Parallel, delayed
import pickle

def app1_segmentation():
    st.title("Customer Segmentation")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    cluster_model = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write("### Dataset Head:")
        st.write(df.head())

        # Handle Missing Values Efficiently
        missing_data_strategy = st.selectbox("Choose how to handle missing data", ["Impute with Mean/Median", "Impute with Mode", "Drop rows", "Drop columns"])

        if missing_data_strategy == "Drop rows":
            df.dropna(inplace=True)
        elif missing_data_strategy == "Drop columns":
            df.dropna(axis=1, inplace=True)
        else:
            num_columns = df.select_dtypes(include=['number']).columns
            if not num_columns.empty:
                num_imputer = SimpleImputer(strategy='median')
                df[num_columns] = num_imputer.fit_transform(df[num_columns])

            cat_columns = df.select_dtypes(include=['object']).columns
            if not cat_columns.empty:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

        # Identify numerical and categorical columns dynamically
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Feature selection
        features = st.multiselect("Select columns for segmentation", df.columns, default=df.columns.tolist())

        if features:
            selected_data = df[features].copy()

            # Encode categorical columns only if they exist
            categorical_cols_in_selected = [col for col in categorical_columns if col in selected_data.columns]
            if categorical_cols_in_selected:
                encoder = OrdinalEncoder()
                selected_data[categorical_cols_in_selected] = encoder.fit_transform(selected_data[categorical_cols_in_selected])

            # Scale numerical columns
            numerical_cols_in_selected = [col for col in numerical_columns if col in selected_data.columns]
            if numerical_cols_in_selected:
                scaler = MinMaxScaler()
                selected_data[numerical_cols_in_selected] = scaler.fit_transform(selected_data[numerical_cols_in_selected])

            # Convert data to NumPy array
            data_matrix = selected_data.to_numpy()

            # Determine clustering method
            has_numerical = any(col in numerical_columns for col in features)
            has_categorical = any(col in categorical_columns for col in features)

            if has_numerical and has_categorical:
                clustering_method = KPrototypes
            elif has_categorical:
                clustering_method = KModes
            else:
                clustering_method = KMeans

            # Optimize Elbow Method Execution
            def compute_wcss(i):
                if clustering_method == KMeans:
                    model = clustering_method(n_clusters=i, n_init=1, random_state=42)
                    model.fit(data_matrix)
                    return model.inertia_
                else:
                    model = clustering_method(n_clusters=i, init='Cao', n_init=1, verbose=0)
                    if has_numerical and has_categorical:
                        model.fit(data_matrix, categorical=[features.index(col) for col in categorical_cols_in_selected])
                    else:
                        model.fit(data_matrix)
                    return model.cost_

            wcss = Parallel(n_jobs=-1)(delayed(compute_wcss)(i) for i in range(1, 7))

            # Plot Elbow Method
            st.write("### Elbow Method for Optimal Clusters")
            fig, ax = plt.subplots()
            ax.plot(range(1, 7), wcss, marker='o', linestyle='--')
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS")
            ax.set_title("Elbow Method for Optimal Clusters")
            st.pyplot(fig)

            # Get user-defined number of clusters
            num_clusters = st.number_input("Select the optimal number of clusters", min_value=1, max_value=10, value=3, step=1)

            # Fit chosen clustering model
            if clustering_method == KMeans:
                model = KMeans(n_clusters=num_clusters, n_init=1, random_state=42)
            else:
                model = clustering_method(n_clusters=num_clusters, init='Cao', n_init=1, verbose=0)

            if has_numerical and has_categorical:
                df['Cluster'] = model.fit_predict(data_matrix, categorical=[features.index(col) for col in categorical_cols_in_selected])
            else:
                df['Cluster'] = model.fit_predict(data_matrix)

            cluster_model = model

            # Visualization
            st.write("### Segmentation Visualization")
            fig = plt.figure()

            if len(features) == 2:
                sns.scatterplot(x=selected_data[features[0]], y=selected_data[features[1]], hue=df['Cluster'], palette='viridis')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.title("Customer Segmentation (2D)")
                plt.legend()
            elif len(features) == 3:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(selected_data[features[0]], selected_data[features[1]], selected_data[features[2]], c=df['Cluster'], cmap='viridis', edgecolors='k')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                ax.set_title("Customer Segmentation (3D)")
                fig.colorbar(scatter, label="Cluster")
            else:
                st.write("Using TruncatedSVD for visualization as more than 3 features are selected.")
                svd = TruncatedSVD(n_components=3)
                reduced_data = svd.fit_transform(data_matrix)
                df[['PCA1', 'PCA2', 'PCA3']] = reduced_data
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(df['PCA1'], df['PCA2'], df['PCA3'], c=df['Cluster'], cmap='viridis', edgecolors='k')
                ax.set_xlabel("PCA1")
                ax.set_ylabel("PCA2")
                ax.set_zlabel("PCA3")
                ax.set_title("Customer Segmentation (SVD-reduced 3D)")
                fig.colorbar(scatter, label="Cluster")

            st.pyplot(fig)

            # Customer Segmentation Results and Save Functionality
            st.write("### Customer Segmentation Results")
            st.write(df['Cluster'].value_counts())

            if st.button("Save Segmentation Results"):
                df.to_csv('segmented_data.csv', index=False)
                st.success("Segmentation results saved for churn prediction!")

            if st.button("Generate CSV for Download"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📅 Click to Download CSV",
                    data=csv,
                    file_name='segmented_customers.csv',
                    mime='text/csv',
                )

    # -------- Save Cluster Model Button --------
    if st.button("Save Cluster Model from Current Session"):
        try:
            if cluster_model is not None:
                with open("cluster_model.pkl", "wb") as f:
                    pickle.dump(cluster_model, f)
                st.success("Cluster model saved as 'cluster_model.pkl'")
            else:
                st.warning("No cluster model found in memory. Please run segmentation module.")
        except Exception as e:
            st.error(f"Failed to save cluster model: {e}")

if __name__ == "__main__":
    app1_segmentation()