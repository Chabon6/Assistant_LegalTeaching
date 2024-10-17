# LegalTeachingAssistant
一個法律教學助理，用於解析法律文件並提取關鍵條文，結合 Advanced RAG，利用 OpenAI GPT 生成基於法律文本的準確解答與解釋。

This project is designed to parse legal documents and extract key provisions, utilizing advanced RAG (Retrieval-Augmented Generation) combined with OpenAI GPT to generate accurate answers and explanations based on legal texts.

### Features
* Law Name Extraction: Automatically extracts the law name from the provided legal text.
* Keyword-based Text Extraction: Retrieves text blocks between specific legal terms or keywords within a legal document.
* Customizable Prompting: Leverages OpenAI's GPT model to process legal questions and generate detailed explanations.
* Integration with LangChain: Uses LangChain for chaining together different text-processing steps and retrieving data from various sources.
