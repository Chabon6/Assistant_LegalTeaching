import os
import json
import re
import streamlit as st
from langchain.schema import Document  # Langchain 的 Document 類別
from striprtf.striprtf import rtf_to_text  # RTF 轉換
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------------- 自訂系統日誌函式 ----------------------
def log_system(message : str):
    """
    將系統訊息記錄到 Streamlit 的 session_state 中，方便在指定區塊中顯示日誌訊息。
    """
    if "system_logs" not in st.session_state:
        st.session_state["system_logs"] = []
    st.session_state["system_logs"].append(message)
    
# ---------------------- 解析與切塊邏輯 ----------------------
class LawParser:
    """
    解析法律文件的文字內容，並將內容依據不同層級（章、節、目、條）進行分割與標記，
    最後轉換為 Langchain 的 Document 物件，包含文本內容（page_content）及相關元資料（metadata）。
    """
    def __init__(self, text_content : str):
        self.text_content = text_content # 原始文字內容
        self.law_name = self.extract_law_name() # 從文本中抽取法規名稱

    def extract_law_name(self):
        """
        透過 re 抽取文本中的法規名稱，格式為「法規名稱：XXX」。

        return:
        - str: 匹配到的法規名稱，如果沒有匹配則返回空字串。
        """
        pattern = r'法規名稱：(\w+)'
        match = re.search(pattern, self.text_content)
        if match:
            return match.group(1)
        else:
            log_system("未找到法規名稱")
            return None

    def extract_text_between_keywords(self, text : str, start_keyword : str, end_keyword=None)->str:
        """
        從指定文本中，抓取兩個關鍵詞之間的文字塊。
        
        params:
        - text: 要搜索的文本
        - start_keyword: 開始關鍵詞
        - end_keyword: 結束關鍵詞
        - end: 沒有 end_keyword 時填 True

        return:
        - 提取出的文字塊，若未找到匹配，返回 None。
        """
        # 若有結束關鍵詞，則以非貪婪模式匹配 start_keyword 與 end_keyword 之間的文字
        if end_keyword:
            pattern = rf"{re.escape(start_keyword)}(.+?){re.escape(end_keyword)}"

        # 若無結束關鍵詞，則匹配 start_keyword 後的所有文字
        else:
            pattern = rf"{re.escape(start_keyword)}(.+)$"

        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            log_system('未找到文字塊')
            return None

    def extract_provisions_names(self, text : str, provisions : str)->list:
        """
        提取文本中的章節名稱，格式為「第 X 章(or節or目)...」的部分。

        params:
        - text: 要搜索的文本。
        - provisions: 決定要抓章 or 節 or 目 or 條

        return:
        - match: 匹配到的章、節、目、條名稱 list，若無匹配則返回空 list
        """
        if provisions == '條':
            pattern = r'(第 \d+-?\w? 條)'
        else:
            pattern = rf'(第 \w+ {re.escape(provisions)}.+)'
        match = re.findall(pattern, text)
        return match if match else []

    def provisions_split_list(self, text : str, provision : str)->list:
        """
        根據指定的層級（如章、節、目、條），將文本切割成塊，並在每一塊開頭附加索引，用以識別該部分的層級。

        params:
        - text: 要切割的文本。
        - provision: 決定要切割的章 or 節 or 目

        return:
        - splited_provisions_list: 切割後的文本 list
        """
        # 初始化切割後的文本 list
        splited_provisions_list = []

        # 找出文本中符合「第 X 章／節／目／條」格式的部分，並將其作為切割點
        names = self.extract_provisions_names(text, provision)

        # 賦予不同層級的索引，章為 1、節為 2、目為 3、條為 4，以便後續識別不同層級的文本。如果傳入的層級不在字典中，則預設為 0。
        provision_index = {'章': 1, '節': 2, '目': 3, '條': 4}.get(provision, 0)
        
        # 若有找到目前的層級，則開始切割文本
        if names:
            # 保留先前層級已處理過、附加於開頭的索引
            pattern = r'\[.+\]'
            match = re.findall(pattern, text)
            for i, name in enumerate(names):
                # 若不是最後一個章節目，則以當前與下一個章節目為切割點
                if i != len(names) - 1:
                    extract_text = self.extract_text_between_keywords(text=text, start_keyword=name, end_keyword=names[i + 1])
                # 若是最後一個章節目，則以當前章節目為切割點，並讓 end_keyword 保持預設值 None
                else:
                    extract_text = self.extract_text_between_keywords(text=text, start_keyword=name)
                # 將先前的開頭所引與新的索引結合，之後才接上切割後的文本。最後將結果加入切割後的文本 list
                splited_provisions_list.append(f"{match[0] if match else ''} [{provision_index}_{name}] " + extract_text)

        # 如果 names 是空的，表示文本中沒有當前層級的區塊，此時就直接將整段文本作為一個單位加入切割後的文本 list
        else:
            splited_provisions_list.append(text)
        return splited_provisions_list

    def law_split_list(self)->list:
        """
        多層次分割原始法律文件：按章、節、目、條的順序進行切割，最後返回切割後的文本 list。

        return:
        - final_list: 切割後的文本 list
        """
        Chapter_list = self.provisions_split_list(self.text_content, "章")
        Section_list = []
        for splited_text in Chapter_list:
            Section_list += self.provisions_split_list(splited_text, "節")
        Item_list = []
        for splited_text in Section_list:
            Item_list += self.provisions_split_list(splited_text, "目")
        final_list = []
        for splited_text in Item_list:
            final_list += self.provisions_split_list(splited_text, "條")
        return final_list

    def list2doc(self, final_list : list)->list:
        """ 
        將切割後的文本 list 轉換為 Langchain 的 Document 物件，並附加相關元資料。

        params:
        - final_list: 切割後的文本 list

        return:
        - docs: 轉換後的 Document 物件 list
        """
        docs = []
        for i, text in enumerate(final_list):
            try:
                Chapter = re.findall(r'\[1.+?\]', text)[0]
            except:
                Chapter = ''
            try:
                Section = re.findall(r'\[2.+?\]', text)[0]
            except:
                Section = ''
            try:
                Item = re.findall(r'\[3.+?\]', text)[0]
            except:
                Item = ''
            try:
                Article = re.findall(r'\[4.+?\]', text)[0]
            except:
                Article = ''
            
            # 將文本內容與相關元資料轉換為 Document 物件，並加入到 docs list 中
            docs.append(Document(
                page_content=text, 
                metadata={
                    "law_name": self.law_name, 
                    "Chapter": Chapter, 
                    "Section": Section, 
                    "Item": Item, 
                    "Article": Article
                }
            ))
        return docs

# ---------------------- 載入設定檔 ----------------------
def load_config():
    """
    載入設定檔 settings.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'settings.json')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config, config_path

# ---------------------- 檢查並處理新 rtf 檔案 ----------------------
def process_new_rtf_files():
    """
    檢查 rtf 資料夾中是否有新的 rtf 檔案，
    若有則處理並更新向量資料庫，同時更新 settings.json 中的 processed_files
    """
    # 載入設定檔
    config, config_path = load_config()
    rtf_dir = config['path_Law_rtf']
    persist_directory = config['path_Chroma_db']

    # 取得已處理檔案列表，若不存在則預設為空列表
    processed_files = config.get("processed_files", [])

    # 找出所有 rtf 檔案，挑選出尚未處理的檔案
    all_rtf_files = [f for f in os.listdir(rtf_dir) if f.endswith('.rtf')]
    new_files = [f for f in all_rtf_files if f not in processed_files]

    # 若有新檔案，則進行處理
    if new_files:
        log_system("發現新的 rtf 檔案，開始處理...")
        new_docs = []

        # 逐一處理新的 rtf 檔案
        for filename in new_files:
            log_system(f"Processing file: {filename}")

            # 讀取 rtf 檔案內容，並轉換為純文字
            file_path = os.path.join(rtf_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
            text_content = rtf_to_text(rtf_content)

            # 使用 LawParser 函式將純文字內容解析與切塊
            parser = LawParser(text_content)
            final_list = parser.law_split_list()

            # 將切塊後的文本轉換為 Document 物件，並加入到 new_docs 中
            docs = parser.list2doc(final_list)
            new_docs.extend(docs)
            log_system(f"Example: {docs[len(docs) // 2]}...")
            log_system(f"Finished processing {filename}")

        # 初始化 OpenAIEmbeddings 
        gpt_api = os.getenv("OPENAI_API_KEY")
        if gpt_api is None:
            st.error("請先設定 OPENAI_API_KEY 環境變數")
            st.stop()
        embedding_model = OpenAIEmbeddings(openai_api_key=gpt_api)

        # 若向量資料庫已存在於指定資料夾中，則將新的 rtf 更新進去；否則建立新的資料庫
        if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 1:
            chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            chroma_db.add_documents(new_docs)
            log_system("已更新現有的 Chroma 資料庫")
        else:
            chroma_db = Chroma.from_documents(new_docs, embedding_model, persist_directory=persist_directory)
            log_system("已建立新的 Chroma 資料庫")

        # 更新 processed_files 並寫回 settings.json
        processed_files.extend(new_files)
        config["processed_files"] = processed_files
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4) 
    else:
        log_system("無新的 rtf 檔案需要處理.")
