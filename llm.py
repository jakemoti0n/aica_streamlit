from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from config import answer_examples
from dotenv import load_dotenv
import os

load_dotenv()

index_name = 'tax-index'
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단 된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해 주세요.
        사전: {dictionary}
        질문: {{question}}                                     
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain 

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
        
    return llm

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')# 3072
    vectorstore = PineconeVectorStore.from_existing_index(  
        # from_existing_index DB에 있는 데이터를 읽기 위한 함수
        index_name=index_name,
        embedding=embedding
        #Pinecone에 저장해둔 벡터 값들(도큐먼트)을 index_name으로 찾아서,
        #embedding 모델로 변환된 쿼리와 비교 검색할 수 있도록 vectorstore 객체를 생성
    )    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()        

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        # MessagesPlaceholder < (안에값)을 들어오는 마커역할임, 함수와는 약간다름
        # 여기선 챗 프롬프트에 표시값(챗히스토리)을 넣음
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    ) 
    
    return history_aware_retriever 


def get_rag_chain():
    llm = get_llm()
       
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,   
    )
 
    
    history_aware_retriever = get_history_retriever()
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해 주세요."
        "아래에 제공된 문서를 활용해서 답변해 주시고 "
        "답변을 알 수 없다면 모른다고 답변해 주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해 주시고 "
        "2~3 문장 정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, # 여기서 Runuble이란 랭체인 을 말함
        get_session_history, # get~history 함수를 받아온다
        input_messages_key="input", #키값1
        history_messages_key="chat_history", #키값2
        output_messages_key="answer", #키값3
    ).pick('answer')     # 여러개의 키값중 앤서값만 뽑아서 반환하는 헬퍼 메서드
    
    return conversational_rag_chain  


def get_ai_message(user_message):  
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()    
    
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_message = tax_chain.stream( #stream 스트리밍 실행 메서드
        {
            "question": user_message # 유저메시지를 받아옴 퀘스천을 키로 랭체인에전달
        },
        config={
            "configurable": {"session_id": "abc123"} 
            #"abc123" = 그냥 예시 세션 ID. 실제 서비스에서는 
            # 각 유저마다 고유 ID를 넣어야 서로 다른 세션이 구분됨
        },
    )        
    
    return ai_message