import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image
import random

# --- Local Image Directory ---
LOCAL_IMAGE_DIR = "local_images"
st.set_page_config(page_title="Customer Recommendation Engine", layout="wide")
def get_category_images(category_name, count=10):
    """Load images from local folder for a given category"""
    images = []
    category_path = os.path.join(LOCAL_IMAGE_DIR, category_name.lower())
    
    if not os.path.isdir(category_path):
        st.warning(f"No folder found for category '{category_name}' in {LOCAL_IMAGE_DIR}")
        return []
    
    image_files = sorted([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png','.webp','.bmp'))])
    
    for i in range(min(count, len(image_files))):
        image_path = os.path.join(category_path, image_files[i])
        try:
            img = Image.open(image_path)
            images.append(img)
        except:
            st.warning(f"Failed to load image: {image_path}")
    
    return images

# --- Main App ---
def app3_recommendations():
    st.title("🛍️ Customer Recommendation Engine")

    try:
        # Load churn prediction results
        df = pd.read_csv('churn_prediction_results.csv')
        
        st.header("🔍 Identify High-Risk Customers")
        threshold = st.slider("Churn Probability Threshold", 0.0, 1.0, 0.7, 0.05)
        high_risk = df[df['Churn_Probability'] >= threshold]
        
        if high_risk.empty:
            st.warning("No high-risk customers found at this threshold.")
            return
        
        st.success(f"Found {len(high_risk)} high-risk customers")
        
        st.header("⚙️ Recommendation Setup")
        selected_customer = st.selectbox(
            "Select customer to analyze",
            high_risk['CustomerID'] if 'CustomerID' in high_risk.columns else high_risk.index
        )
        
        numerical_features = [col for col in df.select_dtypes(include='number').columns if col != 'CustomerID']
        similarity_features = st.multiselect("Features for similarity analysis", numerical_features, default=['Cluster'] if 'Cluster' in numerical_features else numerical_features[:1])
        
        recommendation_features = [col for col in df.columns if col not in similarity_features]
        recommendation_feature = st.selectbox("Base recommendations on", recommendation_features)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Finding similar customers..."):
                not_churned = df[df['Churn_Prediction'] == 0]
                knn = NearestNeighbors(n_neighbors=5).fit(not_churned[similarity_features])
                customer_data = high_risk[high_risk['CustomerID'] == selected_customer] if 'CustomerID' in high_risk.columns else high_risk.loc[[selected_customer]]
                _, indices = knn.kneighbors(customer_data[similarity_features])
                similar_customers = not_churned.iloc[indices[0]]

                if df[recommendation_feature].dtype == 'object':
                    recommendations = similar_customers[recommendation_feature].value_counts().head(3).index.tolist()
                else:
                    recommendations = [f"Avg {recommendation_feature}: {similar_customers[recommendation_feature].mean():.2f}"]
                
                st.header("🎯 Recommendations")
                all_recommendations = []

                for rec in recommendations:
                    with st.expander(f"**{rec}** (Popular among {len(similar_customers)} similar customers)", expanded=True):
                        images = get_category_images(str(rec), 10)
                        cols = st.columns(10)

                        for i, img in enumerate(images):
                            with cols[i % 10]:
                                st.image(img, use_container_width=True)
                                st.caption("Top-rated product in this category")
                                price = round(random.uniform(19.99, 99.99), 2)
                                st.markdown(f"💰 **Price:** ${price}")

                        all_recommendations.append(rec)

                st.download_button(
                    "📥 Download All Recommendations",
                    data=pd.DataFrame({'Recommendations': all_recommendations}).to_csv(index=False),
                    file_name=f"{selected_customer}_recommendations.csv",
                    mime="text/csv"
                )

    except FileNotFoundError:
        st.error("❌ File 'churn_prediction_results.csv' not found. Please run the churn model first.")

if __name__ == "__main__":
    app3_recommendations()
