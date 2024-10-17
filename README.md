# LegalTeachingAssistant
一個法律教學助理，旨在幫助解析法律文件，提取相關條文，並使用 Advanced RAG 來根據法律文本生成解答和解釋。

This project is a Legal Teaching Assistant, designed to help in parsing legal documents, extracting relevant sections, then using advanced RAG to generate answers and explanations based on legal texts.

### Features
* Law Name Extraction: Automatically extracts the law name from the provided legal text.
* Keyword-based Text Extraction: Retrieves text blocks between specific legal terms or keywords within a legal document.
* Customizable Prompting: Leverages OpenAI's GPT model to process legal questions and generate detailed explanations.
* Integration with LangChain: Uses LangChain for chaining together different text-processing steps and retrieving data from various sources.
