import os, json
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# 從業務邏輯模組引入 process_new_rtf_files 與 log_system 函式
from law_processing import process_new_rtf_files, log_system
import logging

# 設定日誌級別為 INFO
logging.basicConfig(level=logging.INFO)

# ---------------------- 自訂 CSS 與 JS ----------------------
# CSS：利用 Flexbox 建立一個 main-container，將畫面分為上方的聊天對話區與下方的輸入區
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        margin: 0;
    }
    .main-container {
        display: flex;
        flex-direction: column;
    }
    .chat-container {
        flex: 1;
        overflow-y: auto;
        background-color: transparent;
    }
    .input-container {
        background-color: transparent;
        border: none;
    }
    /* 對話氣泡樣式：設定文字為黑色 */
    .chat-bubble {
        margin: 5px 0;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        word-wrap: break-word;
        color: black;
    }
    .user-bubble {
        background-color: #F5F5F5;
        margin-left: auto;
        text-align: left;
        color: black;
    }
    .bot-bubble {
        background-color: #DCDCDC;
        margin-right: auto;
        text-align: left;
        color: black;
    }
    textarea {
        width: 100%;
        min-height: 50px;
        max-height: 150px;
        resize: none;
        overflow-y: auto;
        font-size: 1rem;
        padding: 8px;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- 載入設定檔 ----------------------
def load_config():
    """
    載入 JSON 格式的設定檔
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'settings.json')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

# ---------------------- 載入向量資料庫 ----------------------
@st.cache_resource # 將向量資料庫的載入結果快取起來（因為 Streamlit 每次互動都會重執行整個腳本，但被快取的部分只會運算一次）
def load_vector_store():
    """
    載入向量資料庫 (Chroma)：
      1. 從 config 取得資料庫路徑
      2. 從環境變數讀取 OpenAI API key，若未設定則記錄錯誤並停止執行
      3. 利用 OpenAIEmbeddings 初始化嵌入模型，再嘗試載入現有的 Chroma 資料庫
      4. 若資料庫不存在或內容不足，則顯示錯誤訊息並停止應用程式
    """
    # 設定 OpenAI API key
    config = load_config()
    gpt_api = os.getenv("OPENAI_API_KEY")
    if gpt_api is None:
        log_system("OPENAI_API_KEY 讀取失敗")
        st.error("OPENAI_API_KEY 讀取失敗")
        st.stop()

    # 載入向量資料庫
    persist_directory = config['path_Chroma_db']
    embedding_model = OpenAIEmbeddings(openai_api_key=gpt_api)
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 1: # db 至少要有兩個檔案
        chroma_db = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embedding_model
        )
        log_system("已載入現有的 Chroma 資料庫")
    else:
        log_system("向量資料庫不存在")
        st.error("向量資料庫不存在")
        st.stop()
    return chroma_db, config

# ---------------------- Small to Big ----------------------
def get_targets(unique_docs : list) -> dict:
    """
    從檢索到的 n 個文本，各自找出上一層的章, 節 or 目（放大檢索）作為 LLM 回答的參考目標

    params:
    - unique_docs: 檢索到的文本，每個文本帶有 metadata 

    return:
    - targets: 上一層的章, 節 or 目。 e.g. {'Section': {'[2_第 三 節 股東會]', '[2_第 四 節 董事及董事會]'}}
    """
    targets = {} # 用來存放放大的 metadata
    for doc in unique_docs:
        for metadata_key in ['Item', 'Section', 'Chapter']:
            # 若 metadata 不為空（有上層索引），則將其加入 targets
            if doc.metadata[metadata_key] != '':
                try:
                    targets[metadata_key].add(doc.metadata[metadata_key])
                except:
                    targets[metadata_key] = set()
                    targets[metadata_key].add(doc.metadata[metadata_key])
                break
    return targets

# ---------------------- 生成答案 ----------------------
def generate_answer(question, chroma_db, gpt_api):
    """
    根據使用者輸入的法律問題，從向量資料庫中檢索相關文本塊，並嵌入 Prompt 中，最後利用 LLM 生成答案。
      1. 使用 PromptTemplate 定義回答格式與提示內容，包括角色定位、任務描述以及格式要求
      2. 從 chroma_db 建立檢索器(retriever），並設定檢索應返回的文本塊數量
      3. 自檢索結果中的章（或節／目）標籤，索引出上一層的完整文本內容塊（Small to Big）
      4. 將所有取得的文本塊合併後填入 prompt，呼叫 LLM 生成答案
    
    params:
    - question: 使用者輸入的法律問題
    - chroma_db: 向量資料庫
    - gpt_api: OpenAI API key

    return:
    - result.content: LLM 生成的答案
    """

    prompt_template = PromptTemplate(
        input_variables=["Quation", "Chunks"],
        template='''
                    <rule>你是一位專業的中華民國法律顧問，負責回答使用者的法律諮詢</rule>
                    <task>根據檢索的文本(chunks)回答問題，若你不知道答案就回答"不知道"，不要虛構。此外，應對每一選項附上參考來源，例如基於XX法XX章第X條，該選項正確(或錯誤)。</task>
                    <Quation>{Quation}</Quation>
                    <chunks>{Chunks}</chunks>
                    <format>
                    結論：
                    原因：
                    各選項所對應的完整法條：
                    (A)
                    (B)
                    (C)
                    (D)
                    </format>
                '''
    )

    # 建立檢索器並設定返回最相關的 3 個文本塊
    retriever = chroma_db.as_retriever()
    retriever.search_kwargs = {"k": 3}
    unique_docs = retriever.invoke(question) # 使用者提問 Quation 為檢索參考而非 Prompt
    logging.info("檢索到的文件：%s", unique_docs) # 紀錄檢索到的文件

    # 找出上一層的章, 節 or 目作為 LLM 回答的參考目標
    targets = get_targets(unique_docs)

    # 根據參考目標的章節或目，從資料庫索引出上一層的所有條文，並合併納入 prompt
    total_chunks = ''
    for target in targets.keys():
        for i in targets[target]:
            docs_data = chroma_db.get(where={target: i})
            total_chunks += str(docs_data['documents'])

    # 文本塊與問題填入 prompt，呼叫 LLM 生成答案
    prompt = prompt_template.format(Quation=question, Chunks=total_chunks)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=gpt_api)
    result = llm.invoke(prompt)
    return result.content

# ---------------------- 主程式：建立互動介面 ----------------------
def main():
    """
      1. 設定頁面標題與初始化使用者對話歷史記錄
      2. 呼叫 process_new_rtf_files() 檢查並處理新上傳的 RTF 檔案
      3. 載入向量資料庫與設定檔，取得 OpenAI API 金鑰
      4. 使用自訂結構建立主畫面，包含上方對話區與下方輸入區
      5. 當使用者提交問題時，顯示「正在處理…」訊息，生成回答並更新對話歷史，重新整理頁面以顯示最新對話內容
      6. 在頁面底部顯示系統資訊，如檔案處理與資料庫載入狀態日誌
    """
    st.title("法科解題 Chatbot")
    
    # 在頁面頂部建立一個 placeholder 用來顯示「正在處理…」
    processing_placeholder = st.empty()
    
    # 進入互動介面前先檢查是否有新的 rtf 需要處理
    process_new_rtf_files()

    # 初始化對話歷史記錄（存放於 session_state）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 載入向量資料庫與設定檔，並取得 OpenAI API 金鑰
    chroma_db, config = load_vector_store()
    gpt_api = os.getenv("OPENAI_API_KEY")
    
    # 利用一個 main-container 將對話區與輸入區包裹起來
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # 上方：對話區
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if "user" in chat:
            st.markdown(f"<div class='chat-bubble user-bubble'>{chat['user']}</div>", unsafe_allow_html=True)
        if "assistant" in chat:
            st.markdown(f"<div class='chat-bubble bot-bubble'>{chat['assistant']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 下方：使用者輸入區
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_area("請輸入你的法律問題：", key="user_input")
        submitted = st.form_submit_button("送出")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # 關閉 main-container

    # 當使用者送出問題時，先在頂部顯示「正在處理…」
    if submitted and user_question:
        processing_placeholder.markdown(
            "<div style='text-align: center; font-size: 1.2em;'>正在處理…</div>",
            unsafe_allow_html=True
        )
        answer = generate_answer(user_question, chroma_db, gpt_api)
        st.session_state.chat_history.append({"user": user_question, "assistant": answer})
        processing_placeholder.empty()  # 清除「正在處理…」訊息
        st.rerun()
    
    # 下方：系統資訊（例如檔案處理、資料庫載入狀態等）
    if "system_logs" in st.session_state and st.session_state.system_logs:
        with st.container():
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 系統資訊")
            for log in st.session_state.system_logs:
                st.write(log)

# 啟動主程式
if __name__ == '__main__':
    main()

