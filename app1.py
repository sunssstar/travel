####### 고친것 
# 1. 벡터디비에서 못가져오던거 가져옴 (함수 선언안했었음)
# 2. 파인튜닝 모델도 프롬프트 먹는거 확인 (프롬프트를 다르게 넣을때 답변이 다르게 나옴)
# 3. 프롬프트 수정 


##### 고칠것 
# 1. 더 긴 답변을 원할때 로딩속도 
# 2. 그냥 처리속도 
# 3. 큰글씨와 라이브 텍스트 같이 쓰는법 
# 4. gpt 사용량 자꾸 초기화됨 
# 5. 모델을 골라주세요 중복안되게 나오게 하기 
# 6. 속도체크 부분 지워서 다시 제대로 나오게 하기 
# 7. 이전 나왔던 답변 다시 안나오게 (두번나옴)
# 8. 중간에 모델 바꿨을때 안넘어가는듯 

########################
# 1. 임포트 
########################
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


########################
# 2. 설정 
########################

# 사용량 설정 
MAX_GPT_USAGE = 3
gpt_usage_count = 0
last_reset_date = datetime.date.today()
def check_and_reset_gpt_usage():
    global gpt_usage_count, last_reset_date
    today = datetime.date.today()
    if today > last_reset_date:
        gpt_usage_count = 0  # 사용량을 초기화
        last_reset_date = today
        
# 로깅설정 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 전역 변수 및 클라이언트 초기화
COLLECTION_NAME = "son99_d"
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_store = LangchainQdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))



# 모델 초기화
@st.cache_resource
def load_model():
    model_name = "centwon/ko-gpt-trinity-1.2B-v0.5_v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model
# 모델 로드
tokenizer, model = load_model()


########################
# 3. 큐드란트 클래스
# 반환할 문서 수 조정 
########################

# 3-1. 클래스 정의
class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
    # 3-2. 검색기 + 메타데이터  
    # 바꿀 설정값 : topk : 유사 문장을 얼마나 넣을것인지 
    def retrieve(self, query_vector: List[float], top_k: int = 10) -> List[Document]:
        results = self.client.search(
            collection_name=self.collection_name,
            # 임베딩한 쿼리를 넣어주고 
            query_vector=query_vector,
            # DB에서  top 몇개의 문장을 넣을것인지 
            limit=top_k
        )
        if not results:
                logging.info("Qdrant에서 유사한 쿼리가 없습니다.")
                return []
        documents = []
        for result in results:
            text = result.payload.get('답변', 'N/A')
            metadata = {
                    'id': result.id,
                    '질병 카테고리': result.payload.get('질병_카테고리', 'N/A'),
                    '질병': result.payload.get('질병', 'N/A'),
                    '부서': result.payload.get('부서', 'N/A'),
                    '의도': result.payload.get('의도', 'N/A'),
                    'score': result.score
                }
            # Document 객체 생성 및 리스트에 추가
            documents.append(Document(page_content=text, metadata=metadata))
            return documents    
                
        else: 
            logging.info("유사한 쿼리가 없습니다")
    

    
#  3-3. 검색기 
def simple_search(query: str, top_k: int = 5) -> List[Document]:
    # 쿼리 임베딩 
    query_vector = embeddings.embed_query(query)
    # 객체 생성 
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    documents = retriever.retrieve(query_vector, top_k)

    
    # 설정한 k 개수 만큼 문서 반환 
    if not documents:
        logging.warning("Qdrant에서 검색 결과가 없습니다.")
        return []
    logging.info("DB에서 검색 결과를 가져왔습니다.")
    return documents


########################
# 3. LLM
# 온도 조정
########################

########################
# 3-1. gpt 모델 
def generate_gpt4_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> str:
    # 오류가 나지 않을 경우 
    try:
        start_time = time.time()
        documents = simple_search(query)
        if documents:
            # 첫 번째 문서의 메타데이터를 가져옴
            top_document = documents[0]
            metadata = {
                '질병': top_document.metadata.get('질병', 'N/A'),
                '의도': top_document.metadata.get('의도', 'N/A'),
                '답변': top_document.page_content,
                '유사도': top_document.metadata.get('score', 'N/A')
            }
        # 시스템에게 역할과 규칙을 알려주는 메세지
        system_message = (
        "You are a professional medical assistant designed to provide accurate and concise medical advice to older people in emergencies."
        "Your primary goal is to provide information when you talk about a particular illness or condition mentioned."
        "Use the reference information provided to provide users with an accurate response."
        "Avoid adding unnecessary information, and maintain a simple and direct response."
        "Suitable medical answers must be provided."
        )   
        # 사용자의 질문을 모델에게 전달하는 메세지 
        user_message = (
            f"User Query: {query}\n\n"
            f"- Disease: {metadata.get('질병', 'N/A')}\n"
            f"- Reference Answer: {metadata.get('DB 답변', 'N/A')}\n"
            f"Please provide a clear and concise response addressing the symptoms of {metadata.get('질병', 'N/A')} and the recommended actions."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                # 시스템
                {"role": "system", "content": system_message},
                # 유저
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,  # 변경할 부분
            n=1,  # 생성할 응답 개수
            # 맥스토큰 일부로 삭제
            )
        generated_response = response.choices[0].message.content.strip()
        processing_time = time.time() - start_time
        logging.info(f"GPT-4 응답 생성 완료: 길이={len(generated_response)}")
 
        logging.info('-'*30)
        logging.info('참고한 메타데이터 정보')
        logging.info('-'*30)
        logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
        logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
        logging.info(query)
        logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
        logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")
        return generated_response,"GPT-4",processing_time

            

    # 에러 발생시     
    except Exception as e:
        logging.error(f"GPT-4 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. GPT-4 응답을 생성하는 중에 오류가 발생했습니다." ,'error',0.0



########################
# 3-2. 파인튜닝 모델 
# gpt 처럼 사용가능?? 
# 토큰값 처음 200으로 고정 

def generate_custom_model_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None, max_tokens: int = 35) -> str:
    try:
        documents = simple_search(query)
        if documents:
            # 첫 번째 문서의 메타데이터를 가져옴
            top_document = documents[0]
            metadata = {
                '질병': top_document.metadata.get('질병', 'N/A'),
                '의도': top_document.metadata.get('의도', 'N/A'),
                '답변': top_document.page_content,
                '유사도': top_document.metadata.get('score', 'N/A')
            }
        # 프롬프트 생성
        if max_tokens > 100:  # 토큰 수가 많을 경우
            prompt = (
                f"Question: {query}\n\n"
                "Provide a detailed, professional, and medically accurate response to the question above. "
                "The response should be clear, well-structured, and cover the most relevant aspects of the condition, including symptoms, possible complications, and recommended treatment approaches. "
                "Ensure the response is appropriate for a healthcare context and provides valuable insights for the user. "
                "You are a professional healthcare assistant. Provide a clear, precise, and well-structured response to the user's question. "
                "The response should demonstrate your expertise and understanding of the condition. Avoid repetition and ensure each statement adds value to the user's understanding. "
                "Each paragraph should be separated by two new lines for better readability. "
                "Ensure that each sentence in the response is followed by a newline for better readability.\n\n"  # 문장마다 띄어쓰기 요청
                "답변임:"
            )

        else:  # 토큰 수가 적을 경우
            prompt = (
                f"Question: {query}\n\n"
                f"Provide a brief and concise response about {top_document.metadata.get('질병', 'N/A')}. "
                "Limit the response to the key points only and do not exceed the token limit. "
                "End the response naturally within the given token limit.\n\n"
                "답변임:"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,  
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.3,
            top_p=0.95,
            do_sample=True
        )
        
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        logging.info("파인튜닝된 모델을 사용 중입니다.")
        logging.info('-'*30)
        logging.info('참고한 메타데이터 정보')
        logging.info('-'*30)
        logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
        logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
        logging.info(query)
        logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
        logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")
        
        if "답변임:" in response:
            response = response.split("답변임:")[-1].strip()
        return response, "Custom Model",processing_time

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",'error',0.0
        



########################
# 3-3. 사용량 처리 
def generate_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None, max_tokens: int = 35) -> Tuple[str, str, float]:
    try:
        check_and_reset_gpt_usage()

        if 'gpt_usage_count' not in st.session_state:
            st.session_state.gpt_usage_count = 0

        if st.session_state.model_selected == "GPT-4" and st.session_state.gpt_usage_count < MAX_GPT_USAGE:
            st.session_state.gpt_usage_count += 1
            response, model, processing_time = generate_gpt4_response(query, context, metadata)
            logging.info(f"GPT-4 사용 횟수: {st.session_state.gpt_usage_count}")
        else:
            response, model, processing_time = generate_custom_model_response(query, context, metadata, max_tokens=max_tokens)
            logging.info(f"Custom 모델을 사용하여 응답 생성")
            if max_tokens == 35:  # 짧은 답변의 경우에만 추가
                    response += " ....(자세히)" 

        return response, model, processing_time

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error", 0.0

########################
# 4. 쿼리 처리
# 쿼리에서 몇개 가져올지 
########################
            
def process_query(query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, str, float]:
    start_time = time.time()
    logging.info(f" 질문 : {query}")
    try:
        # 유사도 기반으로 디비에서 가져오기
        search_results = simple_search(query, 5)
        if not search_results:
            logging.warning("DB 답변 없음 ")
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?"
        best_match = search_results[0]
        context = best_match.page_content
        
        response, model,processing_time = generate_response(query, context, best_match.metadata)
        logging.info(f"(쿼리함수) 처리 시간: {processing_time:.2f}초")
        
        if model == "Custom Model":
            logging.info(f"사용된 토큰 수: {35 if st.session_state.model_selected == 'Custom-200' else 800}")

        return response, model,processing_time
       
    except Exception as e:
        logging.error(f"쿼리 처리 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error", time.time() - start_time



########################
# 5. 스트림릿 페이지 
########################

# 글자 하나씩 출력하는 함수
def display_typing_effect(text):
    output = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        output.markdown(f"<p>{displayed_text}</p>", unsafe_allow_html=True)
        time.sleep(0.05)

def main():
    st.title("의료 상담 챗봇")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "assistant", "content": "무엇을 도와드릴까요?"})

    if 'gpt_usage_count' not in st.session_state:
        st.session_state.gpt_usage_count = 0

    if 'model_selected' not in st.session_state:
        st.session_state.model_selected = "GPT-4"

    if 'show_long_response_option' not in st.session_state:
        st.session_state.show_long_response_option = False

    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    model_option = st.radio("응답 생성 모델을 선택하세요:", ("GPT-4", "Custom Model"), key="model_selection")

    if model_option != st.session_state.model_selected:
        st.session_state.model_selected = model_option
        st.session_state.chat_history.append({"role": "system", "content": f"모델이 {model_option}로 변경되었습니다."})

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        st.session_state.current_query = user_input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("답변 생성 중..."):
            response, model, processing_time = generate_response(query=user_input, max_tokens=35)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
                logging.info(f"사용된 모델: {model}")
                logging.info(f"처리 시간: {processing_time:.2f}초")

            if model == "Custom Model":
                st.session_state.show_long_response_option = True

    if st.session_state.show_long_response_option:
        if st.button("긴 답변을 원하시나요?"):
            with st.spinner("긴 답변 생성 중입니다...생성 시간이 기존보다 더 걸릴수 있습니다"):
                long_response, _, long_processing_time = generate_response(query=st.session_state.current_query, max_tokens=800)
                st.session_state.chat_history[-1] = {"role": "assistant", "content": long_response}
                with st.chat_message("assistant"):
                    st.write(long_response)
                    logging.info(f"사용된 모델: Custom Model (긴 답변)")
                    logging.info(f"처리 시간: {long_processing_time:.2f}초")

            st.session_state.show_long_response_option = False

    st.sidebar.write(f"GPT-4 사용 횟수: {st.session_state.gpt_usage_count}/{MAX_GPT_USAGE}")

if __name__ == "__main__":
    main()