"""
社内ドキュメント RAGチャットボット - Streamlit版
実行: streamlit run rag_app.py
必要: pip install streamlit anthropic numpy
"""

import os
import numpy as np
import streamlit as st
import anthropic

st.set_page_config(
    page_title="社内ドキュメント検索AI",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');
    * { font-family: 'Noto Sans JP', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); color: #e2e8f0; }
    .header-box { background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%); border: 1px solid #2d5a8e; border-radius: 12px; padding: 24px 32px; margin-bottom: 24px; }
    .header-box h1 { color: #60a5fa; font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; }
    .header-box p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
    .doc-card { background: #1e2433; border: 1px solid #2d3748; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }
    .doc-card h4 { color: #60a5fa; margin: 0 0 6px 0; font-size: 0.9rem; }
    .doc-card p { color: #94a3b8; margin: 0; font-size: 0.82rem; line-height: 1.5; }
    .chat-user { background: #1e3a5f; border-radius: 10px 10px 2px 10px; padding: 12px 16px; margin-bottom: 12px; color: #e2e8f0; text-align: right; }
    .chat-ai { background: #1e2433; border: 1px solid #2d3748; border-radius: 10px 10px 10px 2px; padding: 12px 16px; margin-bottom: 12px; color: #e2e8f0; line-height: 1.7; }
    .source-badge { display: inline-block; background: #1e3a5f; border: 1px solid #3b82f6; color: #60a5fa; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; margin-right: 6px; }
    .stat-card { background: #1e2433; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; text-align: center; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #3b82f6; }
    .stat-label { color: #64748b; font-size: 0.8rem; margin-top: 4px; }
    .section-title { color: #64748b; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
    hr { border-color: #2d3748 !important; }
</style>
""", unsafe_allow_html=True)

DOCUMENTS = [
    {"id": "1", "title": "有給休暇申請ルール", "content": "有給休暇は2週間前までに申請が必要です。申請は社内システムから行い、直属の上司の承認が必要です。繁忙期（3月・9月）は原則として有給取得を控えてください。"},
    {"id": "2", "title": "残業申請ルール", "content": "残業は事前申請が原則です。月45時間を超える場合は課長の承認に加え、人事部への報告が必要です。深夜残業（22時以降）は部長承認が必要です。"},
    {"id": "3", "title": "経費精算ルール", "content": "経費精算は発生月の翌月10日までに提出してください。領収書は必須です。交通費は実費精算、接待費は1人あたり5,000円以内が原則です。"},
    {"id": "4", "title": "リモートワーク規定", "content": "リモートワークは週3日まで申請可能です。コアタイムは10時〜15時です。公共Wi-Fiでの業務は禁止されています。"},
    {"id": "5", "title": "システム障害対応フロー", "content": "本番障害発生時は障害管理票を起票してください。関係者にSlackで即時連絡し、30分以内に暫定対応策を報告します。復旧後24時間以内に原因報告書を提出します。"},
    {"id": "6", "title": "情報セキュリティポリシー", "content": "社外への情報持ち出しは原則禁止です。USBメモリの使用は情報システム部の承認が必要です。パスワードは90日ごとに変更してください。"},
]

def build_vocab(documents):
    words = set()
    for doc in documents:
        for word in (doc["title"] + " " + doc["content"]).lower().split():
            if len(word) > 1:
                words.add(word)
    return list(words)

def vectorize(text, vocab):
    words = text.lower().split()
    vec = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        vec[i] = words.count(word)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def retrieve(query, documents, vocab, top_k=2):
    if not vocab:
        return []
    q_vec = vectorize(query, vocab)
    scored = []
    for doc in documents:
        d_vec = vectorize(doc["title"] + " " + doc["content"], vocab)
        score = float(np.dot(q_vec, d_vec))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(doc, score) for score, doc in scored[:top_k] if score > 0]

def generate_answer(query, retrieved, api_key=None):
    if not retrieved:
        context = "関連するドキュメントが見つかりませんでした。"
    else:
        context = "\n\n".join([f"【{doc['title']}】\n{doc['content']}" for doc, _ in retrieved])
    prompt = f"""あなたは社内規定に詳しいアシスタントです。
以下の社内ドキュメントを参照して、質問に日本語で答えてください。

=== 参照ドキュメント ===
{context}
======================

質問: {query}"""
    import requests
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

if "documents" not in st.session_state:
    st.session_state.documents = DOCUMENTS.copy()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vocab" not in st.session_state:
    st.session_state.vocab = build_vocab(DOCUMENTS)

st.markdown("""
<div class="header-box">
    <h1>🔍 社内ドキュメント検索AI</h1>
    <p>RAGチャットボット — 社内規定・マニュアルを自然言語で検索</p>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([2, 1])

with left_col:
    api_key = st.text_input("Anthropic APIキー", type="password", placeholder="sk-ant-...")
    st.markdown("---")
    if st.session_state.chat_history:
        for item in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">💬 {item["query"]}</div>', unsafe_allow_html=True)
            if item["sources"]:
                sources_html = "".join([f'<span class="source-badge">📄 {s}</span>' for s in item["sources"]])
                st.markdown(f"<div style='margin-bottom:8px'>{sources_html}</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="chat-ai">🤖 {item["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:48px 0;color:#475569;"><div style="font-size:2.5rem;margin-bottom:12px">💬</div><p>質問を入力してください</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    query = st.text_input("質問を入力", placeholder="例: 有給休暇の申請期限は？", label_visibility="collapsed")
    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.button("🔍 検索", use_container_width=True)
    with col2:
        if st.button("🗑️ 履歴クリア"):
            st.session_state.chat_history = []
            st.rerun()
    if search_btn and query:
        if not api_key:
            st.error("APIキーを入力してください")
        else:
            with st.spinner("検索中..."):
                retrieved = retrieve(query, st.session_state.documents, st.session_state.vocab)
                answer = generate_answer(query, retrieved, api_key)
                sources = [doc["title"] for doc, _ in retrieved]
                st.session_state.chat_history.append({"query": query, "answer": answer, "sources": sources})
                st.rerun()
    st.markdown('<p class="section-title">クイック質問</p>', unsafe_allow_html=True)
    quick_questions = ["残業45時間を超えた場合は？", "リモートワークは何日まで？", "経費精算の締め切りは？"]
    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                if api_key:
                    with st.spinner("検索中..."):
                        retrieved = retrieve(q, st.session_state.documents, st.session_state.vocab)
                        answer = generate_answer(q, retrieved, api_key)
                        sources = [doc["title"] for doc, _ in retrieved]
                        st.session_state.chat_history.append({"query": q, "answer": answer, "sources": sources})
                        st.rerun()
                else:
                    st.error("APIキーを入力してください")

with right_col:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.documents)}</div><div class="stat-label">登録ドキュメント数</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">登録ドキュメント</p>', unsafe_allow_html=True)
    for doc in st.session_state.documents:
        st.markdown(f'<div class="doc-card"><h4>📄 {doc["title"]}</h4><p>{doc["content"][:60]}...</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-title">ドキュメントを追加</p>', unsafe_allow_html=True)
    new_title = st.text_input("タイトル", placeholder="例: 採用フロー")
    new_content = st.text_area("内容", placeholder="ドキュメントの内容を入力...", height=100)
    if st.button("➕ 追加", use_container_width=True):
        if new_title and new_content:
            new_doc = {"id": str(len(st.session_state.documents) + 1), "title": new_title, "content": new_content}
            st.session_state.documents.append(new_doc)
            st.session_state.vocab = build_vocab(st.session_state.documents)
            st.success(f"「{new_title}」を追加しました")
            st.rerun()
        else:
            st.warning("タイトルと内容を入力してください")