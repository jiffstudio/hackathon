
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# 文章存储位置
PAPER_PERSIST = 'paper_chroma'
# 卡片存储位置
CARD_PERSIST = 'card_chroma'

# openai key
OPENAI_API_KEY = "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"
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


def paper_search_demo():
    """
    基于本地文章知识库与llm做问答
    """
    # 加载存储的文本向量
    vectordb = Chroma(persist_directory=PAPER_PERSIST, embedding_function=embeddings)
    query = "What is a knowledge base?"
    # 相关性搜索，搜索三个最相关的内容
    docs = vectordb.similarity_search(query, k=3)
    print(len(docs))
    print(docs[0])

    llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)

    """chain_type：chain类型 
    stuff: 这种最简单粗暴，会把所有的 document 一次全部传给 llm 模型进行总结。如果document很多的话，势必会报超出最大 token 
    限制的错，所以总结文本的时候一般不会选中这个。 
    map_reduce: 这个方式会先将每个 document 进行总结，最后将所有 document 总结出的结果再进行一次总结。 
    refine: 这种方式会先总结第一个 document，然后在将第一个 document 总结出的内容和第二个 document 一起发给 llm 模型在进行总结，以此类推。这种方式的好处就是在总结后一个 document 的时候，会带着前一个的 document 
    进行总结，给需要总结的 document 添加了上下文，增加了总结内容的连贯性。 这种一般不会用在总结的 chain 上，而是会用在问答的 chain 上，他其实是一种搜索答案的匹配方式。首先你要给出一个问题，他会根据问题给每个 
    document 计算一个这个 document 能回答这个问题的概率分数，然后找到分数最高的那个 document ，在通过把这个 document 转化为问题的 prompt 的一部分（问题+document）发送给 llm 
    模型，最后 llm 模型返回具体答案。 """

    chain = load_qa_chain(llm, chain_type="refine")
    chain.run(input_documents=docs, question=query)


def card_search_demo():
    """
    查抽认卡
    """
    # 加载存储的文本向量
    vectordb = Chroma(persist_directory=CARD_PERSIST, embedding_function=embeddings)
    query = "复杂KBQA任务的定义"
    docs = vectordb.similarity_search(query, k=3)
    print(len(docs))
    print(docs[0])


if __name__ == '__main__':
    card_store_demo()
    card_search_demo()