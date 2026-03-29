from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

#　LLMの設定
llm = OllamaLLM(model="llama3.2")

# 会話履歴を保存するリスト
chat_history = []

# プロンプトテンプレート
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

# Chainの作成
chain = prompt | llm

def chat(question):
    # Chainを実行
    response = chain.invoke({
        "chat_history": chat_history,
         "question": question,
    })

    # 会話履歴を追加
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

# 会話してみる
print(chat("私の名前は翔平です"))
print(chat("私の名前はなんですか？"))

# 会話してみる
print(chat("私の好きな色は青です"))
print(chat("私の好きな色は何ですか？"))