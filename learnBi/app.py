import streamlit as st
from mycomponent import mycomponent
value = mycomponent(my_input_value="hello there")
st.write("Received", value)