
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


def load_pdf(pdf_path):
    loader = UnstructuredFileLoader(pdf_path)
    doc = loader.load()
    return doc


def load_directory(path, file):
    loader = DirectoryLoader(path, glob="**/"+file)
    doc = loader.load()
    return doc


# 加载测试pdf
pdf_path = 'pdf/A Survey on Complex Knowledge Base Question Answering.pdf'
doc = load_pdf(pdf_path)
print(f'initial: There are {len(doc[0].page_content)} characters in your document')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

# TODO 这里只是示例 实际情况是无需在导入文件时全部拆分，而是一段一段处理

split_documents = text_splitter.split_documents(doc)
print(f'split documents: {len(split_documents)}')

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 文本向量
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# 向量数据库
persist_directory = 'chroma'
vector_store = Chroma.from_documents(split_documents, embeddings, persist_directory=persist_directory)
# 持久化存储
vector_store.persist()

# 加载存储的数据
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
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

