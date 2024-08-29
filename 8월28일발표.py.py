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
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import pandas as pd


#################
# 1. 설정
#################
load_dotenv()
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

# 사이드바 설정 함수
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
        if st.session_state.page != "main":
            if choice == "챗봇":
                st.session_state.page = "second"
            elif choice == "병원&약국":
                st.session_state.page = "third"
            elif choice == "페이지3":
                st.session_state.page = "page3"

def main_page():
    background_image = '사진/효자손.png'  # 메인 페이지 배경 이미지 설정

    # 배경색과 이미지를 함께 설정
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #1187cf; /* 배경색 설정 */
            background-image: url("data:image/png;base64,{get_base64(background_image)}"); /* 투명 이미지 오버레이 */
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
            bottom: 290px;
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

# 글자 하나씩 출력하는 함수
def display_typing_effect(text):
    output = st.empty()  # 빈 요소 생성
    displayed_text = ""  # 출력된 텍스트를 저장할 변수
    for char in text:
        displayed_text += char
        output.markdown(f"<p>{displayed_text}</p>", unsafe_allow_html=True)
        time.sleep(0.05)  # 출력 사이에 지연 시간 추가
        
        
def chat_interface():
    # 배경 이미지 설정
    background_image = '사진/002.png'
    basic_background(background_image)
    
    # 모델 설명
    # CSS 스타일을 사용하여 전체 텍스트 크기를 25px로 설정
    st.markdown("""
        <style>
        .stTitle, .stButton button, .stRadio label, .stChatMessage p {
            font-size: 25px !important;
        }
        .stRadio label {
        font-size: 40px !important;  /* 라디오 버튼 레이블 크기 조정 */
    }
        </style>
    """, unsafe_allow_html=True)

    # 1. 라디오 버튼 선택 인터페이스
    if 'model_option' not in st.session_state:
        st.session_state.model_option = None  # 초기값을 None으로 설정

    # 라디오 버튼 선택
    selected_model = st.radio(
        "사용할 모델을 선택하세요:",
        ("GPT-4", "Custom Model"),
        index=None  # 초기 선택을 None으로 설정
    )

    # 2. 모델 선택이 완료되면 챗봇이 선택된 모델을 알려줌
    if selected_model:
        st.session_state.model_option = selected_model
        st.success(f"{st.session_state.model_option} 모델이 선택되었습니다!")

        # 대화 내역 초기화
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "assistant", "content": "무엇을 도와드릴까요?"})

        # 대화 내역 표시
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user", avatar="사진/user.png"):
                    st.write(chat["content"])
            elif chat["role"] == "assistant":
                with st.chat_message("assistant", avatar="사진/chatbot.png"):
                    st.write(chat["content"])

        # 사용자 입력 필드
        user_input = st.chat_input("질문을 입력하세요...")

        # 사용자 질문 처리
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="사진/user.png"):
                st.write(user_input)

            with st.spinner("답변 생성 중..."):
                if st.session_state.model_option == "GPT-4":
                    response, model, processing_time = generate_response(user_input, context=None, metadata=None)
                else:
                    response, model, processing_time = generate_custom_model_response(user_input, context=None, metadata=None)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="사진/chatbot.png"):
                display_typing_effect(response)  # 타이핑 효과로 응답 출력

def hospital_pharmacy_page():
    
hospital_df = pd.read_csv('병원.csv')
pharmacy_df = pd.read_csv('약국.csv')

    # 배경 이미지를 base64로 인코딩하는 함수
def hospital_pharmacy_page():
    # 스트림릿 페이지 설정

    st.markdown(f"""
        <style>

        }}
        .button-style {{
            display: inline-block;
            font-size: 24px;
            font-weight: bold;
            margin: 10px;
            padding: 15px 30px;
            border-radius: 10px;
            text-align: center;
            color: white;
            cursor: pointer;
        }}
        .hospital-button {{
            background-color: lightblue;
        }}
        .pharmacy-button {{
            background-color: red;
        }}
        .all-button {{
            background-color: gray;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "전체 보기"

    # 병원, 약국, 전체 보기 버튼 생성
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("병원", key="hospital", help="병원 마커 표시"):
            st.session_state.selected_option = "병원"
    with col2:
        if st.button("약국", key="pharmacy", help="약국 마커 표시"):
            st.session_state.selected_option = "약국"
    with col3:
        if st.button("전체 보기", key="all", help="병원과 약국 마커 모두 표시"):
            st.session_state.selected_option = "전체 보기"

    col1, col2 = st.columns([2, 2])

    # 지도에 병원과 약국 마커 추가
    with col1:
        m = folium.Map(location=[35.160522, 129.1619484], zoom_start=15)

        if st.session_state.selected_option == "병원":
            # 병원 마커만 표시
            for idx, row in hospital_df.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=row['요양기관명'],
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
        elif st.session_state.selected_option == "약국":
            # 약국 마커만 표시
            for idx, row in pharmacy_df.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=row['요양기관명'],
                    icon=folium.Icon(color='red', icon='plus-sign')
                ).add_to(m)
        else:
            # 전체 보기: 병원과 약국 모두 표시
            for idx, row in hospital_df.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=row['요양기관명'],
                    icon=folium.Icon(color='blue', icon='plus-sign')
                ).add_to(m)
            for idx, row in pharmacy_df.iterrows():
                folium.Marker(
                    location=[row['좌표(Y)'], row['좌표(X)']],
                    popup=row['요양기관명'],
                    icon=folium.Icon(color='red', icon='remove-sign')
                ).add_to(m)

        output = st_folium(m, width=700, height=500)

    # 마커 클릭시 정보 출력
    with col2:
        if output and 'last_object_clicked' in output:
            clicked_location = output['last_object_clicked']
            if clicked_location:
                # 클릭된 마커의 위도와 경도를 가져옴
                clicked_lat = clicked_location['lat']
                clicked_lng = clicked_location['lng']

                # 병원 정보에서 해당 위치를 찾음
                hospital_selected = hospital_df[
                    (hospital_df['좌표(Y)'] == clicked_lat) & 
                    (hospital_df['좌표(X)'] == clicked_lng)
                ]

                # 약국 정보에서 해당 위치를 찾음
                pharmacy_selected = pharmacy_df[
                    (pharmacy_df['좌표(Y)'] == clicked_lat) & 
                    (pharmacy_df['좌표(X)'] == clicked_lng)
                ]

                # 병원 정보 출력
                if not hospital_selected.empty:
                    st.header(f"**병원:** {hospital_selected.iloc[0]['요양기관명']}")
                    st.subheader(f"**종별코드명:** {hospital_selected.iloc[0]['종별코드명']}")
                    st.subheader(f"**주소:** {hospital_selected.iloc[0]['주소']}")
                    st.subheader(f"**전화번호:** {hospital_selected.iloc[0]['전화번호']}")
                    if pd.notna(hospital_selected.iloc[0]['병원홈페이지']):
                        st.subheader(f"**병원 홈페이지:** {hospital_selected.iloc[0]['병원홈페이지']}")

                # 약국 정보 출력
                if not pharmacy_selected.empty:
                    st.header(f"**약국:** {pharmacy_selected.iloc[0]['요양기관명']}")
                    st.subheader(f"**주소:** {pharmacy_selected.iloc[0]['주소']}")
                    st.subheader(f"**전화번호:** {pharmacy_selected.iloc[0]['전화번호']}")


def thrid_page():
    st.title("페이지3")
    st.write("여기에 페이지3의 내용을 표시합니다.")

# 페이지 전환
if 'page' not in st.session_state:
    st.session_state.page = "main"

sidebar_menu()



if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "second":
    chat_interface()
elif st.session_state.page == "third":
    hospital_pharmacy_page()
elif st.session_state.page == "page3":
    thrid_page()