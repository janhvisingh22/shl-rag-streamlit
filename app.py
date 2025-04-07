import streamlit as st
from recommend import get_top_k, generate_response

st.title("SHL Assessment Recommender")
query = st.text_input("Enter candidate job requirement:")

if st.button("Get Recommendations"):
    if query.strip():
        top_df = get_top_k(query)
        response = generate_response(top_df)
        st.subheader("Recommended Assessments")
        st.write(response)
