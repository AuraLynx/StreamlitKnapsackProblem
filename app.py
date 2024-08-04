import streamlit as st
import numpy as np
import pandas as pd
import pulp
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, LpStatus


def init_page() -> None:
    st.title("ナップザック問題を解く")

def data() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"item": '牛乳', "volume": 1, "value": 200, "max_items": 1},
            {"item": "おにぎり", "volume": 0.9, "value": 75, "max_items": 5},
            {"item": "サンドイッチ", "volume": 1, "value": 150, "max_items": 2},
            {"item": "麦茶", "volume": 0.5, "value": 90, "max_items": 3},
        ]
    )
    return df


def main():
    init_page()

    W = st.number_input(
        "ナップザックの容量",
        min_value=1.0, 
        max_value=10.0, 
        value=5.0, 
        step=1.0,
    )
    st.subheader('アイテム、容量、値段、最大容量の表')
    edited_df = st.data_editor(data(), num_rows="dynamic")
    lt = edited_df['item'].tolist()
    w = edited_df['volume'].tolist()
    v = edited_df['value'].tolist()
    max_items = edited_df['max_items'].tolist()

    problem = pulp.LpProblem(sense=pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", lowBound=0, cat='Integer') for i in range(len(w))]

    problem += pulp.lpDot(v, x)
    problem += pulp.lpDot(w, x) <= W

    for i in range(len(x)):
        problem += x[i] <= max_items[i]

    if st.button("最適化計算実行"):
        status = problem.solve()
        st.write("状態", pulp.LpStatus[status])
        result_df = pd.DataFrame(
            columns=['item', 'count']
        )
        tmp = [xs.value() for xs in x]
        for i, a in enumerate(tmp):
            tmp_df = pd.DataFrame(
                {
                    'item': [lt[i]],
                    'count': [a]
                }
            )
            result_df = pd.concat([result_df, tmp_df], ignore_index=True)
        st.dataframe(result_df)
        st.write("最大価値", problem.objective.value())


if __name__ == '__main__':
    main()
