
from revChatGPT.V3 import Chatbot
import os
import logging
import platform

import streamlit as st
import pandas as pd
import numpy as np

import openai

from streamlit_chat import message


def rev_gpt():
    """ gpt的封装，速度好像比原API快一点 """
    os.environ["OPENAI_API_KEY"] = "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"
    chatbot = Chatbot(api_key="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", proxy="http://127.0.0.1:7890")
    chatbot.engine = 'gpt-3.5-turbo'
    print("Chatbot: ")
    for data in chatbot.ask_stream(
            "hello"
    ):
        message = data
        print(message, end="", flush=True)


#  -------------------------------  chat_bot  -----------------------------------------

# 申请的api_key
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


def generate_response(content):
    """ 能用但是gpt3.5速度很慢 """
    # 把用户输入的消息加入到 prompts 中
    st.session_state['prompts'].append({"role": "user", "content": content})
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=st.session_state['prompts'],
        max_tokens=1024,
        temperature=0.6
    )
    message = completion.choices[0].message.content
    print('message: ', message)
    return message


def get_prompts():
    """ 自定义prompt获取 """
    file_name = 'prompt/english_friend.txt'
    return open(file_name, 'r', encoding='utf-8').read()


# 重置聊天界面
def end_click():
    st.session_state['prompts'] = [{"role": "system", "content": get_prompts()}]
    st.session_state['past'] = []
    st.session_state['generated'] = []
    st.session_state['user'] = ""


# 处理聊天按钮点击事件
def chat_click():
    if st.session_state['user'] != '':
        # 获取用户输入的消息
        chat_input = st.session_state['user']
        # 调用 ChatGPT API 生成回答
        output = generate_response(chat_input)
        # 把生成的回答和用户输入的消息存储到 session_state 中
        st.session_state['past'].append(chat_input)
        st.session_state['generated'].append(output)
        st.session_state['prompts'].append({"role": "assistant", "content": output})
        st.session_state['user'] = ""


st.markdown("#### 我是ChatGPT聊天机器人,我可以回答您的任何问题！")

# 如果没有 prompts 提示词，就初始化
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = [{"role": "system", "content": get_prompts()}]
# 如果没有 generated 生成内容，就初始化
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# 如果没有 past 用户历史输入，就初始化
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 显示用户输入框 key是user 输入内容会存入 user session中
user_input = st.text_input("请输入您的问题:", key='user')
# 显示聊天和重置按钮
chat_button = st.button("发送", on_click=chat_click)
end_button = st.button("重置", on_click=end_click)

# 显示 ChatBot 的回答和用户的输入
if st.session_state['generated']:
    # 倒序遍历已经生成的回答和用户的输入
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        # 显示用户的输入
        print(st.session_state['past'][i])
        print(st.session_state['generated'][i])
        # TODO message 模块会报错 -- 不过现在好像也不需要对话形式了？
        # message(st.session_state['past'][i], is_user=True, key=str(i) + '_past')
        # message(st.session_state['generated'][i], key=str(i) + '_generated')
        st.markdown(st.session_state['past'][i])
        st.markdown(st.session_state['generated'][i])

