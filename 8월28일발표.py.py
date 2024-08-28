import streamlit as st
import os
import base64
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant as LangchainQdrant
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from openai import OpenAI as OpenAIClient
import logging
import time
import datetime
from streamlit_option_menu import option_menu

#################
# 1. 설정
#################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COLLECTION_NAME = "son99_d"
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_store = LangchainQdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

MAX_GPT_USAGE = 3
gpt_usage_count = 0
last_reset_date = datetime.date.today()

@st.cache_resource
def load_model():
    model_name = "centwon/ko-gpt-trinity-1.2B-v0.5_v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model

tokenizer, model = load_model()

#################
# 2. 디비 검색 
#################

class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query_vector: List[float], top_k: int = 20) -> List[Document]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        documents = []
        for result in results:
            if result.score >= 0.7:  # 유사도 0.7 이상인 경우에만 추가
                documents.append(Document(
                    page_content=result.payload['답변'],
                    metadata={
                        'id': result.id,
                        'score': result.score,
                        **result.payload
                    }
                ))
        return documents

def simple_search(query: str, top_k: int = 5) -> List[Document]:
    query_vector = embeddings.embed_query(query)
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    return retriever.retrieve(query_vector, top_k)

    # Qdrant에서 검색 결과 가져오기
    qdrant_results = retriever.retrieve(query_vector, top_k)
    
    if not qdrant_results:
        logging.warning("Qdrant에서 검색 결과가 없습니다.")
        return []
    
    # Qdrant 검색 결과로부터 문서 생성
    documents = create_documents_from_qdrant_results(qdrant_results)
    
    return documents

#################
# 3. 응답 생성
#################

def generate_gpt4_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> str:
    try:
        system_message = (
            "You are an expert medical assistant designed to support seniors during medical emergencies "
            "while traveling. Your primary role is to provide accurate and clear medical advice based on the user's "
            "symptoms or questions about diseases. When the user asks about a disease or symptom, "
            "show the related metadata, such as the disease and intent, and then provide the best advice based on "
            "the retrieved information. Please respond in Korean."
        )

        user_message = (
            f"User Query: {query}\n\n"
            f"Context: {context}\n\n"
            f"Metadata:\n"
            f"- Category: {metadata.get('질병 카테고리', 'N/A') if metadata else 'N/A'}\n"
            f"- Disease: {metadata.get('질병', 'N/A') if metadata else 'N/A'}\n"
            f"- Department: {metadata.get('부서', 'N/A') if metadata else 'N/A'}\n"
            f"- Intent: {metadata.get('의도', 'N/A') if metadata else 'N/A'}\n"
            f"- Score: {metadata.get('score', 'N/A') if metadata else 'N/A'}\n"
            f"- Reference Answer: {metadata.get('답변', 'N/A') if metadata else 'N/A'}"
        )

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,
            n=1,
            stop=None,
        )
        generated_response = response.choices[0].message.content.strip()
        logging.info(f"GPT-4 응답 생성 완료: 길이={len(generated_response)}")

        if metadata:
            logging.info('-'*30)
            logging.info('gpt 모델 사용중')
            logging.info('답변에 관한 정보입니다')
            logging.info('-'*30)
            logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
            logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
            logging.info(f"유사도 (score): {metadata.get('score', 'N/A')}")

        return generated_response

    except Exception as e:
        logging.error(f"GPT-4 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. GPT-4 응답을 생성하는 중에 오류가 발생했습니다."

def generate_custom_model_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> Tuple[str, str, float]:
    try:
        prompt = f"질문: {query}\n\n컨텍스트: {context or ''}\n\n메타데이터: {metadata or {}}\n\n답변:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        max_length = 400
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            temperature=0.5,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        
        if metadata:
            logging.info('-'*30)
            logging.info('파인튜닝 모델 사용중')
            logging.info('답변에 관한 정보입니다')
            logging.info('-'*30)
            logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
            logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
            logging.info(f"유사도 (score): {metadata.get('score', 'N/A')}")
        
        logging.info("파인튜닝된 모델을 사용 중입니다.")
        
        return response.split("답변:")[-1].strip(), "Custom Model", processing_time
    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error", 0.0


def generate_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> Tuple[str, str, float]:
    global gpt_usage_count
    check_and_reset_gpt_usage()
    try:
        if gpt_usage_count < MAX_GPT_USAGE:
            gpt_usage_count += 1
            response = generate_gpt4_response(query, context, metadata)
            model = "GPT-4"
        else:
            notice = "무료모델로 전환합니다(유료모델 3회제한). 시간이 조금 더 걸릴 수 있습니다."
            logging.warning(notice)
            response, model, _ = generate_custom_model_response(query, context, metadata)
            model = "Custom Model"
        
        processing_time = time.time() - time.time()
        return response, model, processing_time
    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error", 0.0


def process_query(query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, str, float]:
    start_time = time.time()
    logging.info(f"처리 중인 쿼리: {query}")

    try:
        # 유사도 기반 검색 수행
        search_results = simple_search(query, 5)
        if not search_results or search_results[0].metadata.get('score', 0) < 0.7:
            logging.warning("유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?")
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?", "No Model", time.time() - start_time

        best_match = search_results[0]
        context = best_match.page_content

        # GPT 모델 사용 여부 확인 및 응답 생성
        response, model = generate_response(query, context, best_match.metadata)

        processing_time = time.time() - start_time
        logging.info(f"처리 시간: {processing_time:.2f}초")

        # 메타데이터 정보와 유사도 로그 출력
        if best_match.metadata:
            logging.info('-'*30)
            logging.info('검색된 메타데이터 정보:')
            logging.info(f" 질병 : {best_match.metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {best_match.metadata.get('의도', 'N/A')}")
            logging.info(f"유사도 (score): {best_match.metadata.get('score', 'N/A')}")
            logging.info('-'*30)

        return response, model, processing_time

    except Exception as e:
        logging.error(f"쿼리 처리 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error", time.time() - start_time

def check_and_reset_gpt_usage():
    global gpt_usage_count, last_reset_date
    today = datetime.date.today()
    if today > last_reset_date:
        gpt_usage_count = 0
        last_reset_date = today

#################
# 4. UI 설정
#################

def sidebar_menu():
    with st.sidebar:
        choice = option_menu("Menu", ["챗봇", "병원&약국", "페이지3"],
                             icons=['bi bi-robot', 'bi bi-capsule', ''],
                             menu_icon="app-indicator", default_index=0,
                             styles={
                                 "container": {"padding": "4!important", "background-color": "#fafafa"},
                                 "icon": {"color": "black", "font-size": "25px"},
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#fafafa"},
                                 "nav-link-selected": {"background-color": "#08c7b4"},
                             })
    return choice

def main():
    choice = sidebar_menu()

    if choice == "챗봇":
        chat_interface()
    elif choice == "병원&약국":
        hospital_pharmacy_page()

def chat_interface():
    st.title("의료 정보 챗봇")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "assistant", "content": "무엇을 도와드릴까요?"})

    # 모델 선택 버튼
    model_option = st.radio("응답 생성 모델을 선택하세요:", ("GPT-4", "Custom Model"))

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("답변 생성 중..."):
            if model_option == "GPT-4":
                response, model, processing_time = generate_response(user_input, context=None, metadata=None)
            else:
                response, model, processing_time = generate_custom_model_response(user_input, context=None, metadata=None)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)


def hospital_pharmacy_page():
    st.title("병원 & 약국 정보")
    st.write("여기에 병원과 약국 정보를 표시합니다.")

if __name__ == "__main__":
    main()
