
import streamlit as st
import streamlit.components.v1 as components
import random

st.set_page_config(page_title="可编辑脑图", layout="wide")
p = open("mindMap.html", encoding="utf-8")

components.html(p.read(), height=1000, width=1800, scrolling=True)
