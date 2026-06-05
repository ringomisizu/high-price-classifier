import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
model = joblib.load("high_price_model.pkl")
tfidf = joblib.load("tfidf.pkl")
manual_cols = joblib.load("manual_feature_columns.pkl")
def get_kishu(name):
    if "茶碗" in name:
        return "茶碗"
    if "壺" in name or "壷" in name:
        return "壺"
    if "湯呑" in name or "ぐい呑" in name or "酒杯" in name or "盃" in name:
        return "酒器"
    if "徳利" in name:
        return "徳利"
    if "花瓶" in name or "花入" in name or "花器" in name or "花生" in name:
        return "花器"
    if "皿" in name:
        return "皿"
    if "鉢" in name:
        return "鉢"
    return "その他"
def make_manual_features(text):
    data = {col: 0 for col in manual_cols}
    flags = {
        "人間国宝フラグ": "人間国宝",
        "最上位フラグ": "最上位",
        "共箱フラグ": "共箱",
        "共布フラグ": "共布",
        "栞フラグ": "栞",
        "美品フラグ": "美品",
        "壺フラグ": "壺|壷",
        "茶碗フラグ": "茶碗",
        "皿フラグ": "皿",
        "徳利フラグ": "徳利",
        "本人フラグ": "本人",
        "個展フラグ": "個展",
        "窯変フラグ": "窯変",
        "十二代三輪休雪フラグ": "十二代三輪休雪",
        "酒井田柿右衛門フラグ": "酒井田柿右衛門|柿右衛門",
        "濁手フラグ": "濁手",
        "白萩_flag": "白萩",
        "灰被_flag": "灰被",
        "群青_flag": "群青",
        "即中斎_flag": "即中斎",
        "鵬雲斎_flag": "鵬雲斎",
    }
    for col, pattern in flags.items():
        if col in data:
            data[col] = int(any(w in text for w in pattern.split("|")))
    if "付属品完備" in data:
        data["付属品完備"] = int(data.get("共箱フラグ", 0) and data.get("共布フラグ", 0) and data.get("栞フラグ", 0))
    if "共箱共布セット" in data:
        data["共箱共布セット"] = int(data.get("共箱フラグ", 0) and data.get("共布フラグ", 0))
    if "柿右衛門_濁手" in data:
        data["柿右衛門_濁手"] = int(data.get("酒井田柿右衛門フラグ", 0) and data.get("濁手フラグ", 0))
    artist = ""
    if "【" in text and "】" in text:
        artist = text.split("【")[1].split("】")[0]
    kishu = get_kishu(text)
    for col in [f"作家_{artist}", f"器種_{kishu}", f"作家器種_{artist}_{kishu}"]:
        if col in data:
            data[col] = 1
    return pd.DataFrame([data])[manual_cols]









  
st.title("高額商品判定AI")
text = st.text_input("商品名を入力")
cost = st.number_input("売上原価", min_value=0, value=0, step=1000)
expected_sales = st.number_input("想定売上", min_value=0, value=0, step=1000)
if st.button("判定"):
    X_text = tfidf.transform([text])
    X_manual = make_manual_features(text)
    X_input = hstack([X_text, X_manual])
    proba = model.predict_proba(X_input)[0][1]
    st.metric("高額確率", f"{proba:.1%}")
    if proba >= 0.45:
        st.success("高額候補です")
    else:
        st.info("高額候補ではありません")
    if expected_sales > 0 and cost > 0:
        profit = expected_sales - cost
        expected_profit = proba * profit
        st.metric("利益", f"{profit:,.0f}円")
        st.metric("期待利益", f"{expected_profit:,.0f}円")
    st.subheader("判定理由")
    feature_names = list(tfidf.get_feature_names_out()) + manual_cols
    values = X_input.toarray()[0]
    contributions = values * model.coef_[0]
    reason_df = pd.DataFrame({
        "特徴量": feature_names,
        "寄与度": contributions
    })
    reason_df = reason_df[reason_df["寄与度"] > 0].sort_values("寄与度", ascending=False).head(10)
    st.dataframe(reason_df)


    st.subheader("最終判断者に確認する観点")

    reason_list = str(reason_df["特徴量"].tolist())
    has_point = False

    if "共箱" in reason_list:
        st.write("・箱書の内容確認")
        has_point = True

    if "壺" in reason_list:
        st.write("・器種の市場人気確認")
        has_point = True

    if "徳田八十吉" in reason_list:
        st.write("・本人作・代作品の確認")
        has_point = True

    if not has_point:
        st.write("・作家名、付属品、状態、市場人気の確認")