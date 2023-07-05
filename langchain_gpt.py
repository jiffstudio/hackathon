
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import openai

# 文章存储位置
PAPER_PERSIST = 'paper_chroma'
# 卡片存储位置
CARD_PERSIST = 'card_chroma'

# openai key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 文本向量
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def load_pdf(pdf_path):
    """
    加载特定内容
    """
    loader = UnstructuredFileLoader(pdf_path)
    doc = loader.load()
    print(doc)
    return doc


def load_directory(path, file):
    """
    读取某路径下某种类型文件或全部文件
    """
    loader = DirectoryLoader(path, glob="**/"+file)
    doc = loader.load()
    return doc


def paper_chroma(split_docs):
    """
    :param split_docs: 文章分段document list
    :return: 向量化存储分段文章内容
    """
    # 向量数据库
    vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=PAPER_PERSIST)
    # 持久化存储
    vector_store.persist()


def card_chroma(cards_list, paper_name):
    """
    :param cards_list: cards文本list
    :param paper_name: cards所属文章路径，用来标记source
    :return:生成卡片对应的chroma
    """
    documents = []
    for card in cards_list:
        new_doc = Document(page_content=card, metadata={'source': paper_name})
        documents.append(new_doc)
    # 向量数据库--card
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=CARD_PERSIST)
    # 持久化存储
    vector_store.persist()


def paper_store_demo():
    """
    阅读文段分割与向量化示例 TODO 这里只是示例 实际情况是无需在导入文件时全部拆分，而是一段一段处理
    """
    # 加载测试pdf
    pdf_path = 'pdf/A Survey on Complex Knowledge Base Question Answering.pdf'
    doc = load_pdf(pdf_path)
    print(f'initial: There are {len(doc[0].page_content)} characters in your document')

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )

    # 文本分割
    split_documents = text_splitter.split_documents(doc)
    print(f'split documents: {len(split_documents)}')
    # 向量化
    paper_chroma(split_documents)


def card_store_demo():
    """
    card存储示例
    """
    # 系统生成的问答抽认卡示例
    test_card = "问题：什么是复杂KBQA任务？答案：复杂的知识库问答（Complex KBQA）任务要求使用知识库中的事实来回答复杂的自然语言问题。" \
                "与简单KBQA不同，复杂KBQA需要处理多跳关系或实体聚合，以获取正确答案。它涉及在知识库中进行深入推理和跨实体关联，以解决更复杂的问题。"
    # 用户自制的笔记卡片？
    test_card_2 = "论文第二部分给出复杂KBQA定义，并说明了数据集构建和有效性评估方式。第三部分给出两种主流KBQA方法。"

    cards = [test_card, test_card_2]
    pdf_path = 'pdf/A Survey on Complex Knowledge Base Question Answering.pdf'
    print('cards: ', cards)
    card_chroma(cards, pdf_path)


def paper_search_demo(que):
    """
    基于本地文章知识库与llm做问答，
    """
    # 加载存储的文本向量
    vectordb = Chroma(persist_directory=PAPER_PERSIST, embedding_function=embeddings)
    query = que
    # 相关性搜索，搜索三个最相关的内容
    docs = vectordb.similarity_search_with_score(query, k=3)
    print(docs[0])
    if docs[0][1] >= 0.5:
        print('相似度低，本地知识库可能不含当前提问内容')
    llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

    """chain_type：chain类型 
    stuff: 这种最简单粗暴，会把所有的 document 一次全部传给 llm 模型进行总结。如果document很多的话，势必会报超出最大 token 
    限制的错，所以总结文本的时候一般不会选中这个。 
    map_reduce: 这个方式会先将每个 document 进行总结，最后将所有 document 总结出的结果再进行一次总结。 
    refine: 这种方式会先总结第一个 document，然后在将第一个 document 总结出的内容和第二个 document 一起发给 llm 模型在进行总结，以此类推。这种方式的好处就是在总结后一个 document 的时候，会带着前一个的 document 
    进行总结，给需要总结的 document 添加了上下文，增加了总结内容的连贯性。 这种一般不会用在总结的 chain 上，而是会用在问答的 chain 上，他其实是一种搜索答案的匹配方式。首先你要给出一个问题，他会根据问题给每个 
    document 计算一个这个 document 能回答这个问题的概率分数，然后找到分数最高的那个 document ，在通过把这个 document 转化为问题的 prompt 的一部分（问题+document）发送给 llm 
    模型，最后 llm 模型返回具体答案。 """

    chain = load_qa_chain(llm, chain_type="refine")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(result['output_text'])
    return result['output_text']


def card_search_demo(que):
    """
    查抽认卡
    """
    # 加载存储的文本向量
    vectordb = Chroma(persist_directory=CARD_PERSIST, embedding_function=embeddings)
    query = que
    docs = vectordb.similarity_search_with_score(query, k=3)
    # 余弦相似度形式 docs为tuple的列表 docs[0]为(Document, sim)，所以docs[0][1]输出余弦距离，0-1
    print(docs[0])
    if docs[0][1] >= 0.5:
        print('相似度低，卡片可能未存储有关问题')
        return -1
    # 有对应内容，整合llm给出答案即可
    llm = OpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="refine")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(result['output_text'])
    return result['output_text']


def paper_abstract(text):
    """ 特定段落文本摘要 """
    prompt = [{"role": "system", "content": open('prompt/abstract.txt', 'r', encoding='utf-8').read()},
              {"role": "user", "content": text}]
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt,
        max_tokens=1024,
        temperature=0.4
    )
    message = completion.choices[0].message.content
    print('message: ', message)
    return message


if __name__ == '__main__':
    # card_store_demo()
    # res = card_search_demo('haha')
    # if res != -1:
    #     print('基于抽认卡得到了答案')
    # else:
    #     paper_search_demo("What is a knowledge base?")
    paper_abstract("法律文本以及其他自然语言文本数据，如科学文献、新闻文章或社交媒体，在 "
                   "互联网和专门系统中呈指数级增长。与其他文本数据不同，法律文本在句子或各种文章之间包含法律特定的单词、短语、问题、概念和因素的严格逻辑联系。这些都是为了帮助人们在特定情况下进行正确的论证，避免歧义。不幸的是，这也使得法律领域的信息检索和问答变得比其他领域更加复杂。 "
                   "法律领域的信息检索(IR)主要有两种方法[1]:手工知识工程(KE)和自然语言处理(NLP)。在 KE "
                   "方法中，努力将法律专家记忆和分类案例的方式翻译成数据结构和算法，这些数据结构和算法将用于信息检索。虽然这种方法通常会产生很好的结果，但由于构建知识库时的时间和财务成本，很难在实践中应用。相比之下，基于NLP的IR系统更加实用，因为它们被设计用来通过利用NLP技术快速处理数千兆字节的数据。然而，在设计这样的系统时，提出了几个挑战。例如，法律语言中的因素和概念以不同于常见用法的方式应用[2]。因此，为了有效地回答一个法律问题，它必须比较预先发现的相关文章中问题和句子之间的语义联系。")