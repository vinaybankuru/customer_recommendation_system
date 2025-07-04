import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

# Optional: XGBoost or LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#st.set_page_config(layout="centered", page_title="Churn Prediction")
# Set page config with centered layout
def app2_churn():
    #st.set_page_config(layout="centered", page_title="Churn Prediction")
    st.title("Customer Churn Prediction")

    try:
        df = pd.read_csv('segmented_data.csv')
        st.success("Segmented Data Loaded Successfully")
        
        # Display data with controlled size
        with st.expander("View Data Head", expanded=False):
            st.dataframe(df.head(), height=200, width=800)

        st.subheader("Select Features for Churn Prediction")
        all_columns = df.columns.tolist()
        target = st.selectbox("Target Variable (Churn Status)", all_columns)
        features = st.multiselect("Select Feature Columns", 
                                [col for col in all_columns if col != target],
                                default=[col for col in all_columns if col != target and df[col].nunique() > 1][:5])

        if features and target:
            X = df[features].copy()
            y = df[target].copy()

            # Preprocessing
            cat_cols = X.select_dtypes(include=['object']).columns
            if not cat_cols.empty:
                le = LabelEncoder()
                for col in cat_cols:
                    X[col] = le.fit_transform(X[col].astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_scaled, y)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

            st.subheader("Choose a Model")
            model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LightGBM"])

            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_choice == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            elif model_choice == "LightGBM":
                model = LGBMClassifier(random_state=42)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Model Performance Section
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_proba):.2f}")
                
                with st.expander("Classification Report"):
                    st.text(classification_report(y_test, y_pred))

            with col2:
                st.write("### Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_test, y_pred), 
                        annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        annot_kws={"size": 12})
                plt.tight_layout()
                st.pyplot(fig_cm)

            # ROC Curve
            st.write("### ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle='--')
            ax_roc.set_xlabel("False Positive Rate", fontsize=12)
            ax_roc.set_ylabel("True Positive Rate", fontsize=12)
            ax_roc.set_title("ROC Curve", fontsize=14)
            ax_roc.legend(fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_roc)

            # Feature Importance
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                feat_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                feat_importance.head(10).plot(kind='barh', ax=ax_imp)
                ax_imp.set_title("Top 10 Feature Importances", fontsize=14)
                ax_imp.tick_params(axis='both', which='major', labelsize=12)
                plt.tight_layout()
                st.pyplot(fig_imp)

            # Save results
            df['Churn_Prediction'] = model.predict(scaler.transform(X))
            df['Churn_Probability'] = model.predict_proba(scaler.transform(X))[:, 1]
            df.to_csv('churn_prediction_results.csv', index=False)

            if st.button("Save Trained Model"):
                joblib.dump((model, scaler), 'churn_model.pkl')
                st.success("Model saved as churn_model.pkl")

            # SHAP Analysis
            st.subheader("SHAP Interpretability")
            if st.checkbox("Show SHAP Summary"):
                try:
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_train)
                    fig_shap, ax_shap = plt.subplots(figsize=(12, 6))
                    shap.summary_plot(shap_values, features=pd.DataFrame(X_train, columns=features), show=False)
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                except Exception as e:
                    st.warning(f"SHAP not supported: {e}")

    except FileNotFoundError:
        st.warning("Please run the segmentation module first to generate data.")

if __name__ == "__main__":
    app2_churn()