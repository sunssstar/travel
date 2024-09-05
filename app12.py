# 초기 돌아가는 파일 땅땅
########################
# 1. 임포트 
########################

import streamlit as st
import os
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant as LangchainQdrant
from langchain.schema import Document
from openai import OpenAI as OpenAIClient
import logging
import time
import datetime
from streamlit_option_menu import option_menu
from pydub import AudioSegment
import requests
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import pandas as pd
from pydub.utils import which


########################
# 2. 설정 
########################

load_dotenv()

# st.set_page_config(layout="wide")
AudioSegment.converter = which("ffmpeg")
hospital_df = pd.read_csv('csv/병원.csv')
pharmacy_df = pd.read_csv('csv/약국.csv')


# 로깅설정 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 전역 변수 및 클라이언트 초기화
COLLECTION_NAME = "hyoja_son"

# Streamlit secrets에서 API 키 및 URL 가져오기
qdrant_url = st.secrets["qdrant"]["QDRANT_URL"]
qdrant_api_key = st.secrets["qdrant"]["QDRANT_API_KEY"]
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# QdrantClient 초기화
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# HuggingFace Embeddings 모델 설정
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# LangchainQdrant 벡터 스토어 설정
vector_store = LangchainQdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

# OpenAIClient 초기화
openai_client = OpenAIClient(api_key=openai_api_key)
# 모델 초기화
@st.cache_resource
def load_model():
    model_name = "centwon/ko-gpt-trinity-1.2B-v0.5_v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model
# 모델 로드
tokenizer, model = load_model()


    
    
def TTS(api_key, api_url, text, voice, max_length=300):  # max_length를 API 제한에 맞게 설정
    headers = {
        "appKey": api_key,
        "Content-Type": "application/json"
    }

    # 텍스트를 최대 길이로 분할 (300자 기준)
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    audio_files = []

    for chunk in text_chunks:
        data = {
            "text": chunk,
            "lang": "ko-KR",
            "voice": voice,
            "speed": "0.9",
            "sr": "8000",
            "sformat": "wav"
        }
        response = requests.post(api_url, json=data, headers=headers)

        if response.status_code == 200:
            audio_data = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            audio_file = io.BytesIO()
            audio_data.export(audio_file, format="wav")
            audio_file.seek(0)
            audio_files.append(audio_file)
        else:
            st.error(f"API 요청 실패: {response.status_code}, {response.text}")
            return None

    # 여러 오디오 파일을 하나로 병합
    combined_audio = AudioSegment.empty()
    for audio in audio_files:
        audio.seek(0)
        combined_audio += AudioSegment.from_file(audio, format="wav")
    
    final_audio_file = io.BytesIO()
    combined_audio.export(final_audio_file, format="wav")
    final_audio_file.seek(0)
    return final_audio_file



########################
# 3. 큐드란트 클래스
# 하는 일 : DB 에서 특정 유사도 이상의 관련된 문서들을 끌고와서 검색해준다 
# 적용 한 것 : 유사도 검색기 
# 적용 해 볼 것 : 
# 1. 특정 의도를 기반으로 (메타데이터 : 의도 ) 검색하기 
# 2. 앙상블 검색기 
# 3. 답변데이터에서 -> 질문 데이터로 디비를 바꿨을때 답변의 차이 
# 4. 의도를 넣었을때. 넣지 않았을때 의 차이 
########################
# 3-1. 클래스 정의



class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query_vector: List[float], top_k: int = 10) -> List[Document]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            logging.info(f"검색 결과: {results}")

        except Exception as e:
            logging.error(f"Qdrant 검색 중 오류 발생: {str(e)}", exc_info=True)
            return []

        if not results:
            logging.info("Qdrant에서 유사한 쿼리가 없습니다.")
            return []

        documents = []
        for result in results:
            logging.info(f"검색 결과 처리 중: {result}")
            if result.score >= 0.3:
                text = result.payload.get('답변', 'N/A')
                metadata = {
                    'id': result.id,
                    '질문': result.payload.get('질문', 'N/A'),
                    '답변': result.payload.get('답변', 'N/A'),
                    '질병': result.payload.get('질병', 'N/A'),
                    'score': result.score
                }
                documents.append(Document(page_content=text, metadata=metadata))
            else:
                logging.info(f"유사도 기준을 충족하지 않는 결과: score={result.score}")

        if not documents:
            logging.info("유사한 쿼리가 없습니다.")
            return []

        return documents

# 3-3. 검색기 
def simple_search(query: str, top_k: int = 5) -> List[Document]:
    try:
        # 쿼리 임베딩 생성
        query_vector = embeddings.embed_query(query)
        
        # 쿼리 벡터의 유효성을 확인
        if query_vector is None or not query_vector:
            logging.warning("쿼리 벡터가 생성되지 않았습니다.")
            return []

        retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
        documents = retriever.retrieve(query_vector, top_k)

        if not documents:
            logging.warning("Qdrant에서 검색 결과가 없습니다.")
            return []
        
        logging.info("DB에서 검색 결과를 가져왔습니다.")
        return documents

    except Exception as e:
        logging.error(f"검색 중 오류 발생: {str(e)}", exc_info=True)
        return []



    
#  3-3. 검색기 





########################
# 3. LLM
########################



########################
# 3-1. gpt 모델 
#  고칠것 및 피피티 
# 1. 프롬프트/온도를 어떻게 넣느냐에 따라 답변이 달라짐을 확인 
# 2. 맥스 토큰 값을 지정해주고 / 지정해주지 않은 이유 
# 3. 옵션들을 넣어줬을때 달라지는 답변의 차이 
########################
def generate_gpt4_response(query: str) -> Tuple[str, str, float]:
    try:
        # 문서 검색
        documents = simple_search(query)
        if not documents:
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?", "GPT-4", 0.0

        # 검색된 문서에서 메타데이터 추출
        top_document = documents[0]
        metadata = {
            '질병': top_document.metadata.get('질병', 'N/A'),
            '질문': top_document.metadata.get('질문', 'N/A'),
            '답변': top_document.metadata.get('답변', 'N/A'),
            '유사도': top_document.metadata.get('score', 'N/A')
        }

        # 시스템에게 역할과 규칙을 알려주는 메시지
        system_message = (
            "You are a professional medical assistant designed to provide accurate and concise medical advice to older people in emergencies. "
            "Your primary goal is to provide information when you talk about a particular illness or condition mentioned. "
            "Use the reference information provided to provide users with an accurate response. "
            "Avoid adding unnecessary information, and maintain a simple and direct response. "
             f"Your response should only focus on the topic related to {metadata.get('질병', 'N/A')}.\n"
            f"Suitable medical answers must be provided based on {metadata.get('답변', 'N/A')}."
        )
        # 사용자의 질문을 모델에게 전달하는 메시지 
        user_message = (
            f"User Query: {query}\n\n"
            f"- Disease: {metadata.get('질병', 'N/A')}\n"
            f"- Reference Answer: {metadata.get('답변', 'N/A')}\n"
            # f"Please provide a clear and concise response addressing the symptoms of {metadata.get('질병', 'N/A')} and the recommended actions."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            n=1,
        )

        generated_response = response.choices[0].message.content.strip()
        processing_time = time.time() - time.time()
        logging.info(f"GPT-4 응답 생성 완료: 길이={len(generated_response)}")
        
        logging.info('-'*30)
        logging.info('참고한 메타데이터 정보')
        logging.info('-'*30)
        # logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
        # logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
        logging.info(query)
        logging.info(f"질병 : {metadata.get('질병', 'N/A')}")
        logging.info(f"유사한 질문 : {metadata.get('질문', 'N/A')}")
        logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
        logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")
        return generated_response, "GPT-4", processing_time

    except Exception as e:
        logging.error(f"GPT-4 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. GPT-4 응답을 생성하는 중에 오류가 발생했습니다.", "GPT-4", 0.0


########################
# 3-2. 파인튜닝 모델 
# ppt 에 넣을 것 및 고칠것 
# 1. 프롬프트와 토큰값에 따른 답변의 차이 
# 2. 옵션들을 넣어줬을때 달라지는 답변의 차이 
# 제일 관건 : 어떻게 해야 가장 좋은 퀄리티 있는 답변을 받을수 있는가? 
########################
def generate_custom_model_response(query: str, max_tokens: int = 45) -> Tuple[str, str, float]:
    try:
        # 문서 검색
        documents = simple_search(query)
        if not documents:
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?", "Custom Model", 0.0

        # 검색된 문서에서 메타데이터 추출
        top_document = documents[0]
        metadata = {
            '질문': top_document.metadata.get('질문', 'N/A'),
            '답변': top_document.metadata.get('답변', 'N/A'),
            '질병': top_document.metadata.get('질병', 'N/A'),
            '유사도': top_document.metadata.get('score', 'N/A')
        }

        if max_tokens > 120: # 사용 x
            prompt = (
                f"You are a language model that aims to provide a detailed and comprehensive explanation while retaining the original content.\n"
                f"Take the following sentence and explain it in detail, providing additional context or clarification as needed while keeping the original meaning intact:\n"
                f"Original sentence: <{metadata.get('답변', 'N/A')}>\n"
                "Ensure that the explanation is thorough and includes relevant information to help the reader fully understand the topic.\n"
                f"Your response should only focus on the topic related to [food poisoning].\n"
                "Generate a detailed response in Korean:\n"
                "---\n"
                "답변임!!:"
            )
        else:
                prompt = (
                    f"**Answer**: [{metadata.get('답변', 'N/A')}]\n"
                    f"Based on the answer above, write a summarized content about {metadata.get('질병', 'N/A')}. "
                    "The answer should be written in Korean and ensure that it does not exceed the maximum token limit, so the sentences are not cut off.\n"
                    "답변임!!:"
                )


        # 모델 호출
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.time()

        # "식도염"과 "esophagitis"를 토큰화하여 각각의 ID를 가져옵니다.
        bad_words_ids_ko = tokenizer(["식도염"], add_special_tokens=False).input_ids
        bad_words_ids_en = tokenizer(["esophagitis"], add_special_tokens=False).input_ids

        # 두 리스트를 병합하여 최종 bad_words_ids를 만듭니다.
        bad_words_ids = bad_words_ids_ko + bad_words_ids_en

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.4,
            do_sample=True,
            early_stopping=True,
            bad_words_ids=bad_words_ids   
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        logging.info("파인튜닝된 모델을 사용 중입니다.")
        logging.info('-'*30)
        logging.info('참고한 메타데이터 정보')
        logging.info('-'*30)
        # logging.info(f" 질병 : {metadata.get('질병', 'N/A')}")
        # logging.info(f" 의도 : {metadata.get('의도', 'N/A')}")
        logging.info(query)
        logging.info(f"질병 : {metadata.get('질병', 'N/A')}")
        logging.info(f"유사한 질문 : {metadata.get('질문', 'N/A')}")
        logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
        logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")

        if "답변임!!:" in response:
            response = response.split("답변임!!:")[-1].strip()    
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()

        return response, "Custom Model", processing_time

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Custom Model", 0.0







########################
# 5. 스트림릿 페이지 
########################

# 기본 배경 이미지 설정 함수
def basic_background(png_file, opacity=1):
   bin_str = get_base64(png_file)
   page_bg_img = f"""
   <style>
   .stApp {{
       background-image: url("data:image/png;base64,{bin_str}");
       background-size: cover;  /* 이미지가 잘리지 않도록 설정 */
       background-position: center;
       background-repeat: no-repeat;
       width: 100%;
       height: 100vh;
       opacity: {opacity};
       z-index: 0;
       position: fixed;
   }}
   </style>
   """
   st.markdown(page_bg_img, unsafe_allow_html=True)
   

# 이미지를 base64 문자열로 변환하는 함수
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 글자 하나씩 출력하는 함수
def display_typing_effect(text):
    output = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        output.markdown(f"<p>{displayed_text}</p>", unsafe_allow_html=True)
        time.sleep(0.05)

########################
# 5-1. 사이드 바 및 메인
########################

# 사이드바 설정 함수
def sidebar_menu():
    # 메인 페이지가 아닐 때만 사이드바 표시
    if st.session_state.page != "main":
        with st.sidebar:
            choice = option_menu(
                "Menu", ["챗봇", "병원&약국", "응급상황대처법"],
                icons=['bi bi-robot', 'bi bi-capsule', ''],
                menu_icon="app-indicator", default_index=0,
                styles={
                    "container": {"padding": "4!important", "background-color": "#fafafa"},
                    "icon": {"color": "black", "font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#fafafa"},
                    "nav-link-selected": {"background-color": "#08c7b4"},
                }
            )

            if choice == "챗봇":
                st.session_state.page = "chat_interface"
            elif choice == "병원&약국":
                st.session_state.page = "hospital_pharmacy"
            elif choice == "응급상황대처법":
                st.session_state.page = "video"


# 메인 페이지 설정 함수
def main():
    # 세션 상태 초기화
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    if 'model_option' not in st.session_state:
        st.session_state.model_option = "GPT-4"  # 기본값을 "GPT-4"로 설정
    if 'gpt_usage_count' not in st.session_state:
        st.session_state.gpt_usage_count = 10  # GPT 모델 초기 사용 가능 횟수 설정
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 10  # 초기 사용 가능 횟수 설정 (필요 시 제거 가능)

    # 메인 페이지가 아닐 때만 사이드바 메뉴 호출
    if st.session_state.page != "main":
        sidebar_menu()

    # 페이지 상태에 따른 페이지 호출
    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "chat_interface":
        chat_interface()
    elif st.session_state.page == "hospital_pharmacy":
        hospital_pharmacy_page()
    elif st.session_state.page == "video":
        video()
    elif st.session_state.page == "ad_page":
        ad_page()
                
def main_page():
    # 메인 페이지일 때만 배경 이미지와 스타일 적용
    background_image = '사진/효자손.png'  # 메인 페이지 배경 이미지 설정

    # 배경색과 이미지를 함께 설정
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #1187cf;
            background-image: url("data:image/png;base64,{get_base64(background_image)}");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            width: 100%;
            height: 100vh;
            position: fixed;
            z-index: 0;
        }}

        div.stButton > button {{
            position: fixed;
            bottom: 400px;
            left: 50%;
            transform: translateX(-50%);
            background-color:  #FFFFFF;
            color: black;
            padding: 15px 32px;
            font-size: 80px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            z-index: 10;
        }}

        div.stButton > button:hover {{
            background-color: #007BFF;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

    # 버튼 클릭 시 페이지 전환
    if st.button("효자SON 이용하기 :point_right: "):
        st.session_state.page = "chat_interface"

########################
# 5-1. 사이드 바 및 메인
########################

# 사이드바 설정 함수
def sidebar_menu():
    if st.session_state.page != "main":
        with st.sidebar:
            choice = option_menu(
                "Menu", ["챗봇", "병원&약국", "응급상황대처법"],
                icons=['bi bi-robot', 'bi bi-capsule', ''],
                menu_icon="app-indicator", default_index=0,
                styles={
                    "container": {"padding": "4!important", "background-color": "#fafafa"},
                    "icon": {"color": "black", "font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#fafafa"},
                    "nav-link-selected": {"background-color": "#08c7b4"},
                }
            )
            if choice == "챗봇":
                st.session_state.page = "chat_interface"
            elif choice == "병원&약국":
                st.session_state.page = "hospital_pharmacy"
            elif choice == "응급상황대처법":
                st.session_state.page = "video"

def main():
    # 세션 상태 초기화
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.page = "main"
        st.session_state.model_option = "GPT-4"  # 기본값을 "GPT-4"로 설정
        st.session_state.gpt_usage_count = 2  # GPT 모델 초기 사용 가능 횟수 설정
        st.session_state.chat_history = []
        st.session_state.usage_count = 10  # 초기 사용 가능 횟수 설정

    # 메인 페이지가 아닐 때만 사이드바 메뉴 호출
    if st.session_state.page != "main":
        sidebar_menu()

    # 페이지 상태에 따른 페이지 호출
    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "chat_interface":
        chat_interface()
    elif st.session_state.page == "hospital_pharmacy":
        hospital_pharmacy_page()
    elif st.session_state.page == "video":
        video()
    elif st.session_state.page == "ad_page":
        ad_page()

def main_page():
    background_image = '사진/효자손.png'
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #1187cf;
            background-image: url("data:image/png;base64,{get_base64(background_image)}");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            width: 100%;
            height: 100vh;
            position: fixed;
            z-index: 0;
        }}
        div.stButton > button {{
            position: fixed;
            bottom: 400px;
            left: 50%;
            transform: translateX(-50%);
            background-color:  #FFFFFF;
            color: black;
            padding: 15px 32px;
            font-size: 80px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            z-index: 10;
        }}
        div.stButton > button:hover {{
            background-color: #007BFF;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

    if st.button("효자SON 이용하기 :point_right: "):
        st.session_state.page = "chat_interface"

# 사용량 관리 함수
def reduce_usage_count():
    if st.session_state.model_option == "GPT-4":
        if st.session_state.gpt_usage_count > 0:
            st.session_state.gpt_usage_count -= 1
        else:
            show_usage_alert()

def show_usage_alert():
    api_key = os.getenv('TTS_API_KEY')
    if st.session_state.gpt_usage_count <= 0:
        warning_text = "약과가 다 떨어졌어요!\n약과를 충전하시겠나요?\n아니면 제 동생을 불러드릴까요?"
        with st.chat_message("assistant", avatar="사진/아바타2.png"):
            st.write(warning_text)
        if api_key:
            tts_audio = TTS(api_key, os.getenv('TTS_URL'), warning_text, "juwon")
            if tts_audio:
                audio_bytes = tts_audio.read()
                encoded_audio = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
        st.write("GPT 모델 사용이 끝났습니다. 다음 대화를 평가해 주세요:")
        st.feedback(options="thumbs", key="feedback", on_change=handle_feedback)

def handle_feedback():
    st.success("피드백 감사합니다!")

# def display_usage_count():
#     image_path = "사진/약과.png"
#     if st.session_state.model_option == "GPT-4":
#         remaining_count = st.session_state.gpt_usage_count
#         with open(image_path, "rb") as img_file:
#             img_base64 = base64.b64encode(img_file.read()).decode()
#         st.markdown(f"""
#             <div style='text-align: right; font-size: 24px; padding: 10px; background-color: #f0f0f0;'>
#                 <img src="data:image/png;base64,{img_base64}" style="width: 30px; vertical-align: middle; margin-right: 5px;">
#                 남은 약과의 개수 : <strong style="font-size: 24px;">{remaining_count}</strong>
#             </div>
#         """, unsafe_allow_html=True)
##########################################
##########################################
##########################################    
##########################################
##########################################
# Chatbot 인터페이스 함수
##########################################    
def chat_interface():
    background_image = '사진/002.png'
    basic_background(background_image)

    # 화면 상단에 남은 횟수 표시
    # display_usage_count()

    st.markdown("""
        <style>
        .stTitle, .stButton button, .stRadio label, .stChatMessage p {
            font-size: 25px !important;
        }
        .stRadio label {
            font-size: 40px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # TTS API 키 및 URL 할당
    api_key = os.getenv('TTS_API_KEY')
    tts_url = os.getenv('TTS_URL')
    
    # 세션 상태에 voice 변수 추가 및 초기화
    if 'voice' not in st.session_state:
        st.session_state.voice = None  # voice 변수를 기본값으로 초기화
        
    # 라디오 버튼 선택
    selected_model_nickname = st.radio(
        "당신을 도울 효자를 골라주세요",
        ("든든한 맏형", "귀염둥이 막내"),
        index=0 if st.session_state.model_option == "GPT-4" else 1
    )

    if selected_model_nickname == "든든한 맏형":
        selected_model = "GPT-4"
        st.session_state.voice = 'juwon'
    else:
        selected_model = "Custom Model"
        st.session_state.voice = 'doyun'

    # 모델이 변경되었을 때만 상태 업데이트 및 TTS 호출
    if st.session_state.model_option != selected_model:
        st.session_state.model_option = selected_model
        if selected_model == "Custom Model":
            avatar_image = "사진/아바타1.png"
            voice = "doyun"  # Custom Model의 경우 "doyun" 음성 사용
            initial_message = "귀염둥이 막내! 간단한 답변은 제가 설명드릴게요!"
        else:
            avatar_image = "사진/아바타2.png"
            voice = "juwon"  # GPT-4 모델의 경우 "juwon" 음성 사용
            initial_message = "든든한 맏형! 자세한 내용이 필요하시다면 제가 설명드릴게요!"

        with st.chat_message("assistant", avatar=avatar_image):
            st.write(initial_message)

        # TTS 재생
        voice = st.session_state.voice
        play_tts(api_key, tts_url, initial_message, voice)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

    user_input = st.chat_input("예시 문장 : 복통 설사 구토등의 증상이 있어 무슨병일까?")
    response = "" 
    if user_input:
        if st.session_state.model_option == "GPT-4" and st.session_state.gpt_usage_count > 0:
            reduce_usage_count()
        elif st.session_state.model_option == "GPT-4" and st.session_state.gpt_usage_count <= 0:
            show_usage_alert()
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="사진/user.png"):
            st.write(user_input)

        with st.spinner("답변 생성 중..."):
            if selected_model == "GPT-4":
                response, model, processing_time = generate_gpt4_response(user_input)
            else:
                response, model, processing_time = generate_custom_model_response(user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="사진/chatbot.png"):
            display_typing_effect(response)

        # TTS 재생
        vocie = st.session_state.voice
        play_tts(api_key, tts_url, response, vocie)


def play_tts(api_key, tts_url, text, voice):
    """TTS 재생을 위한 함수"""
    if api_key and voice and text:  # 모든 매개변수가 유효한 경우에만 TTS 실행
        try:
            tts_audio = TTS(api_key, tts_url, text, voice)
            if tts_audio:
                audio_bytes = tts_audio.read()
                encoded_audio = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"TTS 에러: {e}")



                
#####################
# 약국 및 병원 정보 
#####################

def hospital_pharmacy_page():
    
# 데이터 로드
        hospital = pd.read_csv('data/병원.csv')
        pharmacy = pd.read_csv('data/약국.csv')

        # 지도 시작 장소 선택
        my_map = folium.Map(
            location=[35.160522, 129.1619484], 
            zoom_start=17
        )

        # 병원 마크 추가
        def add_hospital_markers(map_obj):
            for idx, row in hospital.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=folium.Popup(
                        f"{row['요양기관명']}<br>",
                        max_width=450
                    ),
                    icon=folium.Icon(color='pink', icon='hospital', prefix='fa'),
                    tooltip=row['요양기관명']
                ).add_to(map_obj)

        # 약국 마크 추가
        def add_pharmacy_markers(map_obj):
            for idx, row in pharmacy.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=folium.Popup(
                        f"{row['요양기관명']}<br>",
                        max_width=450
                    ),
                    icon=folium.Icon(color='blue', icon='info-sign'),
                    tooltip=row['요양기관명']

                ).add_to(map_obj)

        # Streamlit 앱 타이틀
        st.title('주변 병원 & 약국 정보를 알려드릴께요! ')

        # 선택 박스 추가
        option = st.selectbox(
            '어떤 곳을 찾고 싶으신가요?',
            ('병원', '약국', '전체')
        )

        # 선택에 따라 마커 추가
        if option == '병원':
            add_hospital_markers(my_map)
        elif option == '약국':
            add_pharmacy_markers(my_map)
        else:  # 전체
            add_hospital_markers(my_map)
            add_pharmacy_markers(my_map)

        # 지도 시각화
        output = st_folium(my_map, width=800, height=600)

        # 클릭된 마커의 정보 표시
        if output:
            clicked_location = output.get('last_object_clicked', None)
            if clicked_location:
                clicked_lat = clicked_location['lat']
                clicked_lng = clicked_location['lng']

                hospital_selected = hospital[
                    (hospital['좌표(Y)'] == clicked_lat) & 
                    (hospital['좌표(X)'] == clicked_lng)
                ]

                pharmacy_selected = pharmacy[
                    (pharmacy['좌표(Y)'] == clicked_lat) & 
                    (pharmacy['좌표(X)'] == clicked_lng)
                ]

                if not hospital_selected.empty:
                    st.header(f"**병원:** {hospital_selected.iloc[0]['요양기관명']}")
                    st.subheader(f"**종별코드명:** {hospital_selected.iloc[0]['종별코드명']}")
                    st.subheader(f"**주소:** {hospital_selected.iloc[0]['주소']}")
                    st.subheader(f"**전화번호:** {hospital_selected.iloc[0]['전화번호']}")

                if not pharmacy_selected.empty:
                    st.header(f"**약국:** {pharmacy_selected.iloc[0]['요양기관명']}")
                    st.subheader(f"**주소:** {pharmacy_selected.iloc[0]['주소']}")
                    st.subheader(f"**전화번호:** {pharmacy_selected.iloc[0]['전화번호']}")


def video():


    # 구역 1: 응급상황 대처 동영상

    st.header("응급상황 대처 방법")
    st.video('https://www.youtube.com/watch?v=HktGyea8zcw')

    # 구역 2: 식중독 대처방법

    st.header("식중독 대처방법")
    st.video('https://www.youtube.com/watch?v=nyC11uFLD28')

    # 구역 3: 건강한 여행을 위한 퇴행성 관절염 환자의 팁

    st.header("관절염 환자의 팁")
    st.video('https://www.youtube.com/watch?v=75rWlyi9lXU&t=1s')

    # 구역 4: 응급상황 대처방법

    st.header("응급상황 대처방법")
    st.video('https://www.youtube.com/watch?v=k3DVIXXmmA0')
# 광고 페이지 함수
def ad_page():
    # 배경 이미지 설정
    background_image = '사진/광고페이지.png'
    basic_background(background_image)
    st.session_state.usage_count = 1  # 광고 보고 충전 시 사용 가능 횟수 증가
    st.session_state.page = "chat_interface"
    st.success('광고 시청으로 횟수가 1회 충전되었습니다.')
    st.experimental_rerun()


if __name__ == "__main__":
    main()

                    