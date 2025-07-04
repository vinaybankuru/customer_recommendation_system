import streamlit as st
from app1 import app1_segmentation
from app2 import app2_churn
from app3 import app3_recommendations

def main():
    st.sidebar.title("Customer Retention System")
    module = st.sidebar.radio("Choose Module", [
        "Customer Segmentation", 
        "Churn Prediction", 
        "Recommendation Engine"])
    
    if module == "Customer Segmentation":
        app1_segmentation()
    elif module == "Churn Prediction":
        app2_churn()
    else:
        app3_recommendations()

if __name__ == "__main__":
    main()