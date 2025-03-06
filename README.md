# LegalTeachingAssistant
一個法科助教，用於解析、拆解法規條文，並利用 Advanced RAG 串接 GPT API，使 LLM 生成基於法律文本的準確解答與解釋。

A Legal AI Assistant designed to analyze and interpret statutory provisions, leveraging Advanced RAG (Retrieval-Augmented Generation) to enhance GPT-powered legal reasoning. This system enables LLMs to generate precise and context-aware legal explanations based on authoritative legal texts.

### Features
* Law Name Extraction: Automatically extracts the law name from the provided legal text.
* Keyword-based Text Extraction: Retrieves text blocks between specific legal terms or keywords within a legal document.
* Customizable Prompting: Leverages OpenAI's GPT model to process legal questions and generate detailed explanations.
* Integration with LangChain: Uses LangChain for chaining together different text-processing steps and retrieving data from various sources.
