from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 埋め込みモデルの設定
embeddings = OllamaEmbeddings(model="llama3.2")

# 文章を登録
texts = [
    "Pythonはシンプルで読みやすいプログラミング言語です。",
    "Oracle DBはエンタープライズ向けのデータベースです。",
    "LangChainはLLMアプリを簡単に作るフレームワークです。",
]

# ChromaDBに保存
vectorstore = Chroma.from_texts(texts, embeddings)
print("文書の登録完了！")

# Retrieverの作成
retriever = vectorstore.as_retriever()

# 検索してみる
docs = retriever.invoke("データベースについて教えて")

print("¥n--- 検索結果 ---")
for doc in docs:
    print(doc.page_content)

# LLMの設定
llm = OllamaLLM(model="llama3.2")

# プロンプトテンプレート
prompt = ChatPromptTemplate.from_template("""
以下の文章を参考に質問に答えてください。

文章；
{context}

質問：{question}
""")

# Chainを繋ぐ
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 実行
response = chain.invoke("データベースについて教えて")
print("¥n--- 回答 ---")
print(response)
