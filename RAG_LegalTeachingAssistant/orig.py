import json
import re
from langchain.schema import Document  # Langchain 的 Document 類別

from striprtf.striprtf import rtf_to_text # RTF 轉換
import os # 檔案操作

from langchain_openai import OpenAIEmbeddings # 文本 Embedding
from langchain_chroma import Chroma # 向量儲存庫
from langchain_openai import ChatOpenAI # OpenAI 的 LLM 模型

from langchain_core.prompts import PromptTemplate # 用於定義 prompt 模板

# ============================== Settings ==============================
# get api key from env
gpt_api = os.getenv("OPENAI_API_KEY") 
if gpt_api is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# 讀取 json 配置文件  # 記得改版本
script_dir = os.path.dirname(os.path.abspath(__file__))  # 取得當前腳本所在目錄
config_path = os.path.join(script_dir, 'config', 'settings.json')  # 設定文件路徑
with open(config_path, 'r', encoding='utf-8') as file:
    config = json.load(file)

# ============================== Split Text Function ==============================
# 定義法規解析器
class LawParser:
    def __init__(self, text_content):
        self.text_content = text_content
        self.law_name = self.extract_law_name()

    def extract_law_name(self):
        """
        提取文本中的法規名稱，格式為「法規名稱：XXX」。

        return:
        - str: 匹配到的法規名稱，如果沒有匹配則返回空字串。
        """
        pattern = r'法規名稱：(\w+)'
        match = re.search(pattern, self.text_content)
        if match:
            return match.group(1)
        else:
            print("未找到法規名稱")
            return None

    def extract_text_between_keywords(self, text, start_keyword, end_keyword, end=False):
        """
        從指定文本中，抓取兩個關鍵詞之間的文字塊。

        params:
        - text: 要搜索的文本。
        - start_keyword: 開始關鍵詞。
        - end_keyword: 結束關鍵詞。
        - end: True or False 沒有end_keyword時填 True。

        return:
        - 提取出的文字塊，若未找到匹配，返回 None。
        """
        if not end:
            pattern = rf"{re.escape(start_keyword)}(.+?){re.escape(end_keyword)}"
        else:
            pattern = rf"{re.escape(start_keyword)}(.+)$"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            print('未找到文字塊')
            return None

    def extract_provisions_names(self, text, provisions):
        """
        提取文本中的章節名稱，格式為「第 X 章(or節or目)...」的部分。

        params:
        - text (str): 要搜索的文本。
        - provisions (str): 決定章or節or目

        return:
        - list: 匹配到的章、節、目名稱list，若無匹配則返回空list。
        """
        if provisions == '條':
            pattern = r'(第 \d+-?\w? 條)'
        else:
            pattern = rf'(第 \w+ {re.escape(provisions)}.+)'
        match = re.findall(pattern, text)
        if match:
            return match
        else:
            return []

    def provisions_split_list(self, text, provision):
        """
        根據章、節、目名稱，將文本分割成list。

        參數:
        text (str): 要分割的文本。
        provision (str): 決定章or節or目

        返回:
        list: 分割後的章、節、目名稱list。
        """
        splited_provisions_list = []
        names = self.extract_provisions_names(text, provision)
        provision_index = {'章': 1, '節': 2, '目': 3, '條': 4}.get(provision, 0)

        if names:
            pattern = r'\[.+\]'
            match = re.findall(pattern, text)

            for i, name in enumerate(names):
                if i != len(names) - 1:
                    extract_text = self.extract_text_between_keywords(text, name, names[i + 1])
                else:
                    extract_text = self.extract_text_between_keywords(text, name, "", end=True)
                splited_provisions_list.append(f"{match[0] if match else ''} [{provision_index}_{name}] " + extract_text)
        else:
            splited_provisions_list.append(text)
        return splited_provisions_list

    def law_split_list(self):
        """
        切出條為單位的chunks。

        返回:
        list: 切分後的章、節、目、條的list。
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

    def list2doc(self, final_list):
        """
        每一條文轉為 Langchain 的 Document。

        返回:
        list: 轉換後的 Document list。
        """
        for i, text in enumerate(final_list):
            source = re.findall(r'\[.+\]', text)[0]
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
            final_list[i] = Document(page_content=text, metadata={"law_name": self.law_name, "Chapter": Chapter, "Section": Section, "Item": Item, "Article": Article})
        return final_list
    
# ============================== Get Text from RTF ==============================
# 準備 rtf 檔及初始化 Document list
file_path = config['path_Law_rtf']
final_docs = []

# 在 file_path 底下遍歷 rtf 檔
for i, filename in enumerate(os.listdir(file_path)):

    if filename.endswith('.rtf'):

        print(f"Processing file: {filename}")

        # 讀取 RTF 文件
        with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as file:
            rtf_content = file.read()
            
        # 轉換為純文本
        text_content = rtf_to_text(rtf_content)

        # 切割法典並轉成document
        parser = LawParser(text_content)
        final_list = parser.law_split_list()
        final_docs.append(parser.list2doc(final_list))
        print(f"Example: {final_docs[i][-1]}")
        print("Finished \n---------------------")

# ============================== RetrievalQA ==============================
# 初始化 OpenAI 的 Embedding 模型
persist_directory = config['path_Chroma_db']

# 有檔案就讀檔案
if len(os.listdir(persist_directory)) > 1:

    # 初始化 OpenAI 的 Embedding 模型
    embedding_model = OpenAIEmbeddings(openai_api_key=gpt_api)

    # 載入儲存在本地的 Chroma 資料庫
    chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    print("已載入Chroma資料庫")

else:

    # 初始化 OpenAI 的 Embedding 模型
    embedding_model = OpenAIEmbeddings(openai_api_key=gpt_api)

    # Embedding 並儲存至向量儲存庫（另存一份到本地）
    chroma_db = Chroma.from_documents(final_docs, embedding_model, persist_directory=persist_directory)
    print("已儲存Chroma資料庫")

# Prompt

# template
prompt_template = PromptTemplate(
                  input_variables = ["Quation", "Chunks"],
                  template = '''
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
                  </format>'''
                  )
# 設定使用者提問
Quation = '''
4 A 股份有限公司（下稱「A 公司」）為一家從事營建的非公開發行公司，該公司近期擬召開董事 會，討論公司投資東南亞事宜。依公司法之規定，下列敘述何者錯誤？
(A) A 公司過半數之董事得以書面記明提議事項及理由，請求董事長召集董事會。其請求提出後 15 日內，董事長不為召開時，過半數之董事得自行召集
(B) A 公司董事會之召集，除章程有較高之規定者外，應於 3 日前通知各董事及監察人
(C) A 公司章程得訂明經全體董事同意，董事就當次董事會議案以書面方式行使其表決權，而不實 際集會
(D) A 公司董事居住國外者，得以書面委託居住國內之其他股東，並向主管機關登記後，經常代理出席董事會
'''

# ============================== Small to Big ==============================
# 初始化檢索器
retriever = chroma_db.as_retriever()
# 設定檢索器返回多個段落
retriever.search_kwargs = {"k": 3}  # 設定檢索結果數量為 3
# 以使用者提問 Quation 為檢索參考而非 Prompt
unique_docs = retriever.invoke(Quation)

# 找出上一層的章or節or目
def get_targets(unique_docs):
    targets = {}
    for doc in unique_docs:
        for metadata_key in ['Item', 'Section', 'Chapter']:
            if doc.metadata[metadata_key] != '':
                try:
                  targets[metadata_key].add(doc.metadata[metadata_key])
                except:
                  targets[metadata_key] = set()
                  targets[metadata_key].add(doc.metadata[metadata_key])
                break
    return targets

targets = get_targets(unique_docs)
# 把上一層的所有條文全部索引出來
total_chunks = ''
for target in targets.keys():
    for i in targets[target]:
        total_chunks += str(chroma_db.get(where={target: i})['documents'])

# 初始化 OpenAI 的 LLM 模型
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=gpt_api)

# 提示詞
prompt = prompt_template.format(Quation=Quation, Chunks=total_chunks)

# 輸出
result = llm.invoke(prompt)

print(result.content)