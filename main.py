from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings,GPT4AllEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
from fastapi import FastAPI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
# from transformers import AutoModel, AutoTokenizer
# from langchain.tools import tool
# from langchain.agents import initialize_agent, AgentType
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import ssl
from pyngrok import ngrok, conf, installer
from langchain.retrievers import EnsembleRetriever
import os
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get(""), #you can put the key here directy
    model="gpt-3.5-turbo-16k",
    temperature=0
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["ALLOW_DANGEROUS_DESERIALIZATION"]="TRUE"
with open('data/luat_lao_dong_nice.txt', 'r', encoding='utf-8') as f:
    data_quydinh = f.read()
with open('data/thai_san2_nice.txt', 'r', encoding='utf-8') as f:
    data_thai_san = f.read()
with open('data/quy_trinh_bao_lanh_nice.txt', 'r', encoding='utf-8') as f:
    data_bao_lanh = f.read()
with open('data/cspl_nice.txt', 'r', encoding='utf-8') as f:
    data_cspl = f.read()
with open('data/cham_cong_nice.txt', 'r', encoding='utf-8') as f:
    data_cham_cong = f.read()
with open('data/boi_thuong_suc_khoe_nice.txt', 'r', encoding='utf-8') as f:
    data_boi_thuong_suc_khoe = f.read()

data_all = data_quydinh+data_thai_san+data_bao_lanh+data_cspl+data_boi_thuong_suc_khoe+data_cham_cong
model_embed = OpenAIEmbeddings(model="text-embedding-3-large")
# documents = data_quydinh
# documents_thai_san = data_thai_san
# documents_bao_lanh = data_bao_lanh
# documents_cspl = data_cspl

documents_all = data_all
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=1318,
    chunk_overlap=35,
    # length_function=len

)
text_splitter_all = CharacterTextSplitter(
    separator=".",
    chunk_size=1318,
    chunk_overlap=35,
    # length_function=len

)
embeddings = model_embed
chunks = text_splitter.split_text(data_quydinh)
chunks_thai_san = text_splitter.split_text(data_thai_san)
chunks_bao_lanh = text_splitter.split_text(data_bao_lanh)
chunks_cham_cong = text_splitter.split_text(data_cham_cong)
chunks_cspl = text_splitter.split_text(data_cspl)
chunks_boi_thuong_suc_khoe = text_splitter.split_text(data_boi_thuong_suc_khoe)
chunks_all = text_splitter_all.split_text(data_all)
# # print(chunks)
# embeddings = model_embed
# db_quy_dinh = FAISS.from_texts(chunks, embeddings)
# db_thai_san = FAISS.from_texts(chunks_thai_san, embeddings)
# db_bao_lanh = FAISS.from_texts(chunks_bao_lanh, embeddings)
# db_cham_cong = FAISS.from_texts(chunks_cham_cong, embeddings)
# db_cspl = FAISS.from_texts(chunks_cspl, embeddings)
# db_boi_thuong_suc_khoe = FAISS.from_texts(chunks_boi_thuong_suc_khoe, embeddings)
# db_all = FAISS.from_texts(chunks_all, embeddings)
#
# db_quy_dinh.save_local("db_quy_dinh")
# db_thai_san.save_local('db_thai_san')
# db_bao_lanh.save_local('db_bao_lanh')
# db_cham_cong.save_local('db_cham_cong')
# db_cspl.save_local('db_cspl')
# db_boi_thuong_suc_khoe.save_local('db_boi_thuong_suc_khoe')
# db_all.save_local('db_all')

db_quy_dinh = FAISS.load_local("db_quy_dinh", embeddings,allow_dangerous_deserialization=True)
db_thai_san = FAISS.load_local("db_thai_san", embeddings,allow_dangerous_deserialization=True)
db_bao_lanh = FAISS.load_local("db_bao_lanh", embeddings,allow_dangerous_deserialization=True)
db_cham_cong = FAISS.load_local("db_cham_cong", embeddings,allow_dangerous_deserialization=True)
db_cspl = FAISS.load_local("db_cspl", embeddings,allow_dangerous_deserialization=True)
# db_care = FAISS.load_local("db_care", embeddings,allow_dangerous_deserialization=True)
db_boi_thuong_suc_khoe = FAISS.load_local("db_boi_thuong_suc_khoe", embeddings,allow_dangerous_deserialization=True)
db_all = FAISS.load_local("db_all", embeddings,allow_dangerous_deserialization=True)
# print(111)
custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng với tư cách là nhân sự công ty.
Nếu thông tin có đề cập, bắt đầu câu trả lời là 'Theo quy định'
Tuyệt đối không được bịa ra câu trả lời. Nếu bạn không biết câu trả lời, chỉ cần nói rằng 'Xin lỗi, tôi không có thông tin.'
Tất cả câu trả lời của bạn đều phải trả lời chi tiết bằng tiếng Việt.

Context: {context}
Question: {question}

"""
retriever_all = db_all.as_retriever(search_kwargs={"k": 6})
retriever_quy_dinh=db_quy_dinh.as_retriever(search_kwargs={"k": 5})
retriever_thai_san=db_thai_san.as_retriever(search_kwargs={"k": 4})
retriever_bao_lanh=db_bao_lanh.as_retriever(search_kwargs={"k": 4})
retriever_cham_cong=db_cham_cong.as_retriever(search_kwargs={"k": 4})
retriever_cspl=db_cspl.as_retriever(search_kwargs={"k": 4})
retriever_boi_thuong_suc_khoe=db_boi_thuong_suc_khoe.as_retriever(search_kwargs={"k": 4})

ensemble_retriever = EnsembleRetriever(retrievers=[retriever_thai_san, retriever_all])
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])
chain_openAI_quy_dinh = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_quy_dinh, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)
chain_openAI_thai_san = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_thai_san, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)
chain_openAI_bao_lanh = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_bao_lanh, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)
chain_openAI_cham_cong = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_cham_cong, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)


chain_openAI_cspl = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_cspl, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)
# chain_openAI_care = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model="gpt-3.5-turbo-16k"),
#     chain_type="stuff",
#     retriever=db_care.as_retriever(search_kwargs={"k": 5}),
#     chain_type_kwargs={'prompt': prompt}
# )
chain_openAI_boi_thuong_suc_khoe = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=EnsembleRetriever(retrievers=[retriever_boi_thuong_suc_khoe, retriever_all]),
    chain_type_kwargs={'prompt': prompt}
)

chain_openAI_all = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
    chain_type="stuff",
    retriever=db_all.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={'prompt': prompt}
)
query = "Nội dung của điều số 24 là gì"
# response = chain_openAI.invoke(query)
# print(response)
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


### Construct retriever ###
retriever = db_quy_dinh.as_retriever(search_kwargs={"k": 5})

### Contextualize question ###
contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập
có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, chỉ cần sửa lại câu hỏi nếu cần và nếu không thì trả lại như cũ."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi.
Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết.
Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
# print(1)
# demo = conversational_rag_chain.invoke(
#     {"input": "điều số 24 quy định gì"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]
# print(demo)
# print(2)
# demo = conversational_rag_chain.invoke(
#     {"input": "câu trước đó tôi hỏi là gì"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]
# print(demo)
def HR(text):
    demo = conversational_rag_chain.invoke(
        {"input": text},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    return demo


from langchain.agents import tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
import re
def check_quy_dinh(text):
    pattern = r'điều\s+\d+|điều số\s+\d+'

    # Tìm tất cả các kết quả phù hợp
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if len(matches) == 0:
        return 0
    else:
        return 1
def context_quy_dinh(text):
    text = text.replace('điều', 'Điều')
    text = text.replace('Điều số', 'Điều')
    # Biểu thức chính quy để tìm cấu trúc 'điều + số'
    pattern = r'điều\s+\d+'

    # Tìm tất cả các kết quả phù hợp
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    context = ''
    if len(matches) > 0:
        for text in matches:
            for chunk in chunks:
                if text in chunk:
                    context = context + '...' + chunk
                    if len(context) > 7000:
                        return context
                    # if len(documents) > 4:
                    #     print(documents)
    return context
def quy_dinh_dieu(question):
    # question = 'điều 2 quy định gì'
    context = context_quy_dinh(question)
    custom_prompt_template = f"""Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng với tư cách là nhân sự công ty.
    Nếu thông tin có đề cập, bắt đầu câu trả lời là 'Theo quy định'
    Tuyệt đối không được bịa ra câu trả lời. Nếu bạn không biết câu trả lời, chỉ cần nói rằng 'Xin lỗi, tôi không có thông tin.'
    Tất cả câu trả lời của bạn đều phải trả lời chi tiết bằng tiếng Việt.

    Context: {context}
    Question: {question}

    """
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    res = llm.invoke(custom_prompt_template)
    return res.content
# # In ra các kết quả tìm được
# for match in matches:
#     print(match)
# check_quy_dinh('khi nào tôi có thể ăn cơm tại điều 8')
@tool
def quy_dinh(query: str):
    """Trả lời các câu hỏi về nội quy lao động"""
    print('1')
    if check_quy_dinh(query) == 1:
        print('switch to dieu so...')
        return quy_dinh_dieu(query)
    else:
        return chain_openAI_quy_dinh.invoke(query)['result']


@tool
def thai_san(query: str):
    """Trả lời tất cả các câu hỏi liên quan đến thai sản, ốm đau, mang thai, sinh con, chính sách trợ cấp nuôi con nhỏ"""
    print('2- thai sản')
    return chain_openAI_thai_san.invoke(query)['result']

@tool
def bao_hiem_suc_khoe(query: str):
    """Trả lời tất cả các câu hỏi liên quan đến bảo hiểm sức khỏe, sotacare, bồi thường sức khỏe"""
    print('3- bảo hiểm')
    return chain_openAI_boi_thuong_suc_khoe.invoke(query)['result']
@tool
def cham_cong(query: str):
    """Chỉ trả lời các câu hỏi về quy trình liên quan đến portal và cách thiết lập mật khẩu phiếu lương"""
    print('4- chấm công')
    return chain_openAI_cham_cong.invoke(query)['result']


@tool
def phuc_loi(query: str):
    """Trả lời tất cả các câu hỏi liên quan đến phúc lợi"""
    print('5- phúc lợi')
    return chain_openAI_cspl.invoke(query)['result']

# def nghi_phep(query: str):
#     """Chỉ trả lời các câu hỏi về việc gia hạn, quyết định"""
#     print('3')
#     return chain_openAI_nghi_phep.invoke(query)

@tool
def bao_lanh(query: str):
    """Trả lời tất cả các câu hỏi liên quan đến bảo lãnh viện phí"""
    print('6')
    return chain_openAI_bao_lanh.invoke(query)['result']
tools = [quy_dinh, thai_san, bao_lanh, cham_cong, bao_hiem_suc_khoe, phuc_loi]
from operator import itemgetter
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser

def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

rendered_tools = render_text_description(tools)
system_prompt = f"""Bạn là trợ lý có quyền truy cập vào bộ công cụ sau. Dưới đây là tên và mô tả cho từng công cụ:
{rendered_tools}

Đưa ra thông tin input của người dùng, trả về name và input của công cụ sẽ sử dụng. Trả về output của bạn dưới dạng blob JSON với các keys 'name' và 'arguments'. Lưu ý arguments là nguyên văn câu hỏi của người dùng"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt | model | JsonOutputParser() | tool_chain
import json


def read_history(user_id, conversation_id):
    filename = 'history.json'

    if not os.path.exists(filename):
        return '',''  # Trả về None nếu tệp không tồn tại

    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    if user_id in data and conversation_id in data[user_id]:
        conversation = data[user_id][conversation_id]
        return conversation['human'], conversation['AI']
    else:
        return '',''  # Trả về None nếu không tìm thấy cuộc trò chuyện


# Hàm để cập nhật lịch sử trò chuyện
def update_history(user_id, conversation_id, question, answer):
    filename = 'history.json'

    # Nếu tệp đã tồn tại, đọc dữ liệu hiện tại
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    else:
        data = {}

    # Cập nhật dữ liệu mới
    if user_id not in data:
        data[user_id] = {}

    data[user_id][conversation_id] = {
        "human": question,
        "AI": answer
    }

    # Ghi dữ liệu cập nhật vào tệp JSON
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def final_tool(query):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    try:
        final_answer = chain.invoke({"input": query})
    except:
        final_answer = llm.invoke(query).content
    return final_answer
import time
def chat_with_history(query, user_id, conversation_id):
    human,ai = read_history(user_id, conversation_id)
    ai = ai[:500]
    prompt = f'''KHÔNG trả lời câu hỏi. Dưới đây là lịch sử trò chuyện và câu hỏi mới nhất của người dùng, có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập
    có thể hiểu được nếu không có lịch sử trò chuyện. Chỉ cần sửa lại câu hỏi nếu cần và nếu không thì trả lại như cũ:
    last_query: {human}
    last_answer: {ai}
    query: {query}
    Trả ra câu hỏi nguyên văn hoặc sửa lại nếu cần.
    new_query:
    '''
    if human !='':
        print('using history...')
        print(prompt)
        start = time.time()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        query = llm.invoke(prompt).content
        end = time.time()
        print('history: ',end - start)
    print("new_query: ",query)
    very_final_answer = final_tool(query)
    update_history(user_id, conversation_id, query, very_final_answer)
    return very_final_answer

# def chat_with_history(query):
#     human,ai = read_history()
#     prompt = f'''
#     Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, chỉ cần định dạng lại nó nếu cần và nếu không thì trả lại như cũ.
#     Ví dụ:
#     last_query: Ngày 30/4 có được nghỉ không?
#     query: Hôm đó được thưởng không
#     new_query: Ngày 30/4 có được thưởng không?
#     ------------------------------------
#     last_query: Điều 29 quy định gì
#     query: Điều 15 thì sao
#     new_query: Điều 15 quy định gì
#     ------------------------------------
#     Nếu câu hỏi không cần lịch sử vẫn có thể hiểu được thì giữ lại nguyên văn:
#     last_query: Điều 28 trong quy định có liên quan đến ngày nghỉ lễ không?
#     query: Điều 15 quy định gì?
#     new_query: Điều 15 quy định gì?
#     -----------------------------------
#     last_query: {human}
#     query: {query}
#     new_query:
#     '''
#     if human !='':
#         print('using history...')
#         print(prompt)
#         start = time.time()
#         query = llm.invoke(prompt).content
#         end = time.time()
#         print('history: ',end - start)
#     print("new_query: ",query)
#     very_final_answer = "final_tool(query)"
#     update_history(query,very_final_answer)
#     return very_final_answer

# def final_tool(query):
#     llm=ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)
#     response1 = chain.invoke({"input": query})
#     return response1['result']
    # print(response1)
    # if 'hông có thông tin' in response1['result']:
    #     response2 = chain_openAI_all(query)
    #     return response2['result']
    # response2 = chain_openAI_all(query)
    # print(response2)
    # if len(response2['result']) > len(response1['result']):
    #     return response2['result']
    # else:
    #     return response1['result']
    # final_prompt =f"""
    # Bạn là nhân sự của công ty sotatek chuyên trả lời các thắc mắc của nhân viên.
    # Question {query}
    # Tham khảo thông tin bên dưới để tổng hợp ra câu trả lời đầy đủ nhất:
    # answer1: {response1['result']}
    # answer2: {response2['result']}
    # Hãy trả lời một cách chi tiết
    # Answer:
    # """
    # response = llm.invoke(final_prompt)
    # return response.content
# original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Example</h1>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;background-size: 100vw 100vh;">- Được nghỉ có lương trong những trường hợp nào</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Làm hỏng laptop công ty cấp thì xử lý thế nào</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Các bước thiết lập mật khẩu phiếu lương</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Được ngủ qua đêm tại công ty không</h2>'
# st.markdown(original_title, unsafe_allow_html=True)
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://intoroigiare.vn/wp-content/uploads/2023/11/background-hinh-nen-powerpoint-dep.jpg");
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;
#     background-repeat: no-repeat;
# }
# </style>
# """
# demox = """
# <h2 style="font-family: serif; color:white; font-size: 15px;">spec : int or Iterable of numbers Controls the number and width of columns to insert. Can be one of: * An integer that specifies the number of columns. All columns have equal width in this case. * An Iterable of numbers (int or float) that specify the relative width of each column. E.g. ``[0.7, 0.3]`` creates two columns where the first one takes up 70% of the available with and the second one takes up 30%. Or ``[1, 2, 3]`` creates three columns where the second one is two times the width of the first one, and the third one is three times that width.</h2>
# """
# # st.write(demox, unsafe_allow_html=True)
# st.markdown(background_image, unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .reportview-container .main .block-container div[data-baseweb="toast"] {
#         background-color: red;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# st.title('Hỏi đáp nội quy')
# with st.form('my_form'):
#     text = st.text_area('Input')
#     print(text)
#     submitted = st.form_submit_button('Run')
#     if submitted:
#         query = text
#
#         start = time.time()
#         st.info(final_tool(query))
#         # st.info(chat_with_history(query))
#         end = time.time()
#         print('tổng: ',end-start)

# col1, col2, col3 = st.columns(3)
# if submitted:
#     query = text
#     llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
#
#     response2 = chain_openAI_all(query)
#     with col1:
#         st.header("Answer1")
#         st.info(response2['result'])
#         st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     response1 = chain.invoke({"input": query})
#     with col2:
#         st.header("Answer2")
#         st.info(response1['result'])
#         st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     if 'hông có thông tin' in response1['result']:
#         with col3:
#             st.header("Answer3")
#             st.info(response2['result'])
#             st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     else:
#         final_prompt = f"""
#         Bạn là nhân sự của công ty sotatek chuyên trả lời các thắc mắc của nhân viên.
#         Question {query}
#         Tham khảo thông tin bên dưới để tổng hợp ra câu trả lời đầy đủ nhất:
#         answer1: {response1['result']}
#         answer2: {response2['result']}
#         Hãy trả lời một cách chi tiết
#         Answer:
#         """
#         response = llm.invoke(final_prompt)
#         with col3:
#             st.header("Answer3")
#             st.info(response.content)
#             st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
        # st.info(final_tool(text))
# def final_tool(query):
#     llm=ChatOpenAI(model="gpt-3.5-turbo-16k")
#     response1 = chain.invoke({"input": query})
#     print(response1)
#     if 'hông có thông tin' in response1['result']:
#         response2 = chain_openAI_all(query)
#         return response2['result']
#     response2 = chain_openAI_all(query)
#     print(response2)
#     final_prompt =f"""
#     Bạn là nhân sự của công ty sotatek chuyên trả lời các thắc mắc của nhân viên.
#     Question {query}
#     Tham khảo thông tin bên dưới để tổng hợp ra câu trả lời đầy đủ nhất:
#     answer1: {response1['result']}
#     answer2: {response2['result']}
#     Hãy trả lời một cách chi tiết
#     Answer:
#     """
#     response = llm.invoke(final_prompt)
#     return response.content
app = FastAPI()
class TTSRequest(BaseModel):
    text: str
    user_id: str
    conversation_id: str
    # model: str

@app.post("/chat")
def chatbot(request: TTSRequest):
    print(request.text)
    # demo = chain.invoke({"input": request.text})
    # demo = final_tool(request.text)
    demo = chat_with_history(request.text, request.user_id, request.conversation_id)
    print(demo)
    return demo


if __name__ == "__main__":
    uvicorn.run(app, port=5000, host='0.0.0.0')

