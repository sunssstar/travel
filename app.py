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

# 로깅설정 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 전역 변수 및 클라이언트 초기화
COLLECTION_NAME = "son99_d"
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_store = LangchainQdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

# 사용량 설정 
MAX_GPT_USAGE = 3

# 모델 초기화
@st.cache_resource
def load_model():
    model_name = "centwon/ko-gpt-trinity-1.2B-v0.5_v3"
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
        documents = []
        for result in results:
            if result.score >= 0.7:
                # 유사도 0.7 이상만 
                metadata = result.payload
                # 추가 메타데이터 필드
                full_metadata = {
                    '질병': metadata.get('질병', 'N/A'),
                    '의도': metadata.get('의도', 'N/A'),
                    '유사도':  result.score,
                    # 이부분 나중에 보기 
                    '답변': metadata.get('답변', 'N/A'),
                }
                
                # Document 생성 수정 ?? 
                documents.append(Document(
                    page_content=metadata.get('답변', 'N/A'),
                    metadata=full_metadata
                ))
        return documents        
    
#  3-3. 검색기 
def simple_search(query: str, top_k: int = 5) -> List[Document]:
    # 쿼리 임베딩 
    query_vector = embeddings.embed_query(query)
    # 객체 생성 
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    qdrant_results = retriever.retrieve(query_vector, top_k)
    # 설정한 k 개수 만큼 문서 반환 
    if not qdrant_results:
        logging.warning("Qdrant에서 검색 결과가 없습니다.")
        return []
    logging.info("DB에서 검색 결과를 가져왔습니다.")
    return qdrant_results


########################
# 3. LLM
# 온도 조정
########################

########################
# 3-1. gpt 모델 
def generate_gpt4_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> str:
    # 오류가 나지 않을 경우 
    try:
        if metadata:
            logging.info('-'*30)
            logging.info('참고한 메타데이터 정보')
            logging.info('-'*30)
            logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
            logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
            logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")
        # 시스템에게 역할과 규칙을 알려주는 메세지
        system_message = (
    "You are an expert medical assistant designed to provide accurate and friendly medical advice to seniors in emergency situations. "
    "Your role is to deliver reliable information based on users' questions about diseases or symptoms, "
    "and to offer appropriate advice based on comprehensive information about the diseases. "
    "selecting the relevant information according to the user's query. "
    "Please explain everything in a clear and accessible manner, ensuring that the information is tailored for senior users. "
    "Responses should be coherent and continuous, effectively delivering the necessary information. "
    "Additionally, if there is a token limit, ensure that the response is completed within that limit."
)       
        # 사용자의 질문을 모델에게 전달하는 메세지 
        user_message = (
            f"User Query: {query}\n\n" # 질문 제공
            f"Context: {context}\n\n" # 맥락제공
            f"- Disease: {metadata.get('질병', 'N/A') if metadata else 'N/A'}\n" # 질병 정보 제공 
            f"- Reference Answer: {metadata.get('답변', 'N/A') if metadata else 'N/A'}" # 참고 답변 제공
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
        logging.info(f"GPT-4 응답 생성 완료: 길이={len(generated_response)}")
        return generated_response

            

    # 에러 발생시     
    except Exception as e:
        logging.error(f"GPT-4 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. GPT-4 응답을 생성하는 중에 오류가 발생했습니다."    



########################
# 3-2. 파인튜닝 모델 
# gpt 처럼 사용가능?? 
# 토큰값 처음 200으로 고정 
# def generate_custom_model_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None, max_tokens: int = 200) -> str:
#     try:
#         # 시스템에게 역할과 규칙을 알려주는 메시지
#         system_message = (
#             "You are an expert medical assistant designed to provide accurate and friendly medical advice to seniors in emergency situations. "
#             "Your role is to deliver reliable information based on users' questions about diseases or symptoms, "
#             "and to offer appropriate advice based on comprehensive information about the diseases. "
#             "selecting the relevant information according to the user's query. "
#             "Please explain everything in a clear and accessible manner, ensuring that the information is tailored for senior users. "
#             "Responses should be coherent and continuous, effectively delivering the necessary information. "
#             "Additionally, if there is a token limit, ensure that the response is completed within that limit."
#         )

#         # 사용자의 질문과 관련된 추가 정보
#         user_message = (
#             f"User Query: {query}\n\n"
#             f"Context: {context or 'N/A'}\n\n"
#             f"- Disease: {metadata.get('질병', 'N/A') if metadata else 'N/A'}\n"
#             f"- Reference Answer: {metadata.get('답변', 'N/A') if metadata else 'N/A'}"
#         )
        
#         # 모델 호출
#         input_ids = tokenizer.encode(system_message + "\n\n" + user_message, return_tensors='pt').to(model.device)
#         response = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=max_tokens,  # 응답의 최대 길이
#             num_return_sequences=1,  # 생성할 응답의 개수
#             no_repeat_ngram_size=2,  # n-gram 중복 방지
#             temperature=0.4,  # 창의성 제어
#             do_sample=True,  # 샘플링 방식
#         )
        

#         # 응답 디코딩
#         # clean_up_tokenization_spaces : 공백 자동처리 
#         response_text = tokenizer.decode(response[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
#         if metadata and '답변' in metadata:
#             reference_answer = metadata['답변']
#             # 필요한 부분만 추출 (여기서는 첫 문장을 예시로)
#             extracted_info = reference_answer.split(". ")[0] + "."
#         else:
#             extracted_info = response_text

#         return extracted_info
#         logging.info("파인튜닝된 모델을 사용 중입니다.")

#     except Exception as e:
#         logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
#         return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

def generate_custom_model_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None, max_tokens: int = 200) -> str:
    try:
        if metadata:
            logging.info('-'*30)
            logging.info('참고한 메타데이터 정보')
            logging.info('-'*30)
            logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
            logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
            logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")
        input_ids = tokenizer.encode(query, return_tensors='pt').to(model.device)
        response = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.4,
            do_sample=True
        )
        response_text = tokenizer.decode(response[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if metadata and '답변' in metadata:
            reference_answer = metadata['답변']
            extracted_info = reference_answer.split(". ")[0] + "."
        else:
            extracted_info = response_text

        logging.info("파인튜닝된 모델을 사용 중입니다.")
        
        return extracted_info

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."


########################
# 3-3. 사용량 처리 
def generate_response(query: str, context: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    global gpt_usage_count
    try:
        if gpt_usage_count < MAX_GPT_USAGE:
            response = generate_gpt4_response(query, context, metadata)
            gpt_usage_count += 1
            logging.info(f"GPT-4 사용 횟수: {gpt_usage_count}")
            return response, "GPT-4"
        else:
            response = generate_custom_model_response(query, context, metadata)
            logging.info(f"Custom 모델을 사용하여 응답 생성")
            return response, "Custom Model"

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Error"

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
        
        response, model = generate_response(query, context, best_match.metadata)
        
        # 메타데이터 로그 출력
        if best_match.metadata:
            logging.info('-'*30)
            logging.info('참고한 메타데이터 정보')
            logging.info('-'*30)
            logging.info(f" 질병 : {best_match.metadata.get('질병', 'N/A')}")
            logging.info(f" 의도 : {best_match.metadata.get('의도', 'N/A')}")
            logging.info(f"참고한 답변 : {best_match.metadata.get('답변', 'N/A')}")
            logging.info(f"유사도 (score): {best_match.metadata.get('유사도', 'N/A')}")
        # 사용한 모델 및 토큰 정보 출력
        logging.info(f"사용한 모델: {model}")
        if model == "Custom Model":
            logging.info(f"사용된 토큰 수: {200 if st.session_state.model_selected == 'Custom-200' else 800}")


        # 처리시간 
        processing_time = time.time() - start_time
        logging.info(f"처리 시간: {processing_time:.2f}초")
        return response, model, processing_time
        
       
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
    st.title("의료 정보 챗봇")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'model_selected' not in st.session_state:
        st.session_state.model_selected = False
        st.session_state.chat_history.append({"role": "assistant", "content": "모델을 골라주세요."})
        
    if 'gpt_usage_count' not in st.session_state:
        st.session_state.gpt_usage_count = 0

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    if not st.session_state.model_selected:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("GPT-4 모델 사용 (3회 제한)"):
                st.session_state.model_selected = "GPT-4"
                st.session_state.chat_history.append({"role": "assistant", "content": "GPT-4 모델을 선택하셨습니다."})

        with col2:
            if st.button("커스텀 모델 사용 (토큰 200)"):
                st.session_state.model_selected = "Custom-200"
                st.session_state.chat_history.append({"role": "assistant", "content": "커스텀 모델(토큰 200)을 선택하셨습니다."})



    if st.session_state.model_selected:
        user_input = st.chat_input("질문을 입력하세요...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("답변 생성 중..."):
                if st.session_state.model_selected == "GPT-4":
                    if st.session_state.gpt_usage_count < MAX_GPT_USAGE:
                        response = generate_gpt4_response(user_input)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})                        
                        display_typing_effect(response)
                        # st.session_state.gpt_usage_count += 1
                        logging.info(f"GPT-4 사용 횟수: {st.session_state.gpt_usage_count}")
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": "GPT-4 사용 횟수 초과. 무료 모델을 사용해 주세요."})

                elif st.session_state.model_selected == "Custom-200":
                    response = generate_custom_model_response(user_input, max_tokens=200)
                    st.session_state.chat_history.append({"role": "assistant", "content": "요약된 답변을 생성합니다"})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    display_typing_effect(response)

                    if st.button("더 긴 답변을 원하세요?"):
                        long_response = generate_custom_model_response(user_input, max_tokens=800)
                        st.session_state.chat_history.append({"role": "assistant", "content": "긴 답변을 생성합니다"})
                        st.session_state.chat_history.append({"role": "assistant", "content": long_response})
                        display_typing_effect(long_response)
                        
                elif st.session_state.model_selected == "Custom-800":
                    response = generate_custom_model_response(user_input, max_tokens=800)
                    st.session_state.chat_history.append({"role": "assistant", "content": "긴 답변을 생성합니다"})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    display_typing_effect(response)
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])

if __name__ == "__main__":
    main()