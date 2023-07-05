import os
import json
from langchain.document_loaders import UnstructuredFileLoader
import streamlit as st


def get_first_card(text):
    messages = [{"role": "user", "content": f'''Imagine you are a Text-to-Card Converter. Your task is to take lengthy pieces of text and break them down into several small, easily digestible cards for the user to read. Each card should encapsulate a focused concept but also need to faithfully replicate the original text, including a title and content. Importantly, the language used in the cards must be in Chinese. Some parts may have formatting issues, please fix them. Below is the original text.
    ---------------------------------
    {text}'''}]
    functions = [
        {
            "name": "get_first_card",
            "description": "Get first card in a given text",
            "parameters": {
                "type": "object",
                "properties": {
                    "card": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title, e.g. Concept of RLHF, keep it blank if not focused enough",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content",
                            },
                        }
                    },
                    "remaining": {
                        "type": "string",
                        "description": "The first 10 words of remaining text that is not included in the first card",
                    },
                },
                "required": ["card", "remaining"],
            },
        }
    ]
    import requests

    url = "https://openai.api2d.net/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer fk205005-4JjeuMSr5qUREGOdRyqpS0pWQ6iAf6sM'
        # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }

    data = {
        "model": "gpt-3.5-turbo-0613",
        "messages": messages,
        "functions": functions,
        "function_call": "auto",
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code", response.status_code)
    print("JSON Response ", response.json())
    return response.json()


st.header("PDF Import and Display")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'docx', 'txt'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    with open(os.path.join("pdf_files", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = os.path.join("pdf_files", uploaded_file.name)
    st.write(file_path)
    loader = UnstructuredFileLoader(file_path, mode="elements")
    docs = loader.load()
    print([doc.page_content for doc in docs])
    text = '\n'.join([doc.page_content for doc in docs])
    print(st.session_state)


    if 'cards' in st.session_state:
        for card in st.session_state.cards:
            st.write(f'#### {card["title"]}\n{card["content"]}')

    if 'remaining' not in st.session_state or len(st.session_state.remaining) > 10:
        if st.button('继续'):
            if 'remaining' not in st.session_state:
                st.session_state.remaining = text
                st.session_state.cards = []

            arguments = json.loads(get_first_card(st.session_state.remaining[:1000])['choices'][0]['message']['function_call']['arguments'])
            st.session_state.remaining = st.session_state.remaining[st.session_state.remaining.find(arguments["remaining"][:4]):]
            st.session_state.cards.append(arguments["card"])