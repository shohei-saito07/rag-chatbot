from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# LLMの設定
llm = OllamaLLM(model="llama3.2")

# プロンプトテンプレートの作成
prompt = ChatPromptTemplate.from_template(
    "あなたは{role}です。質問：{question}"
)

# ChainをつなぐPipeで
chain = prompt | llm

# 実行
response = chain.invoke({
    "role":"料理の専門家",
    "question": "カレーの作り方を教えてください"
})
print(response)