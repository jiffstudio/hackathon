import streamlit as st
from learnBi.mycomponent import mycomponent
value = mycomponent(my_input_value="hello there")
st.write("Received", value)