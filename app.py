import streamlit as st
import numpy as np
import pandas as pd
import pulp
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, LpStatus


def init_page() -> None:
    st.title("ナップザック問題を解くGUI")
    st.write("制約条件を書きましょう")

def data() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"item": '牛乳', "volume": 1, "value": 200, "max_items": 10},
            {"item": "おにぎり", "volume": 1, "value": 200, "max_items": 10},
            {"item": "サンドイッチ", "volume": 1, "value": 200, "max_items": 10},
        ]
    )
    return df


def main():
    init_page()

    lt = ['牛乳', "おにぎり", "サンドイッチ"]
    w = [1, 0.7, 0.5]
    v = [200, 150, 120]
    max_items = [10, 30, 2]
    W = st.number_input(
        "ナップザックの容量",
        min_value=1.0, 
        max_value=50.0, 
        value=1.0, 
        step=1.0,
    )
    edited_df = st.data_editor(data(), num_rows="dynamic")

    problem = pulp.LpProblem(sense=pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", lowBound=0, cat='Integer') for i in range(len(w))]

    problem += pulp.lpDot(v, x)
    problem += pulp.lpDot(w, x) <= W

    for i in range(len(x)):
        problem += x[i] <= max_items[i]

    if st.button("実行"):
        status = problem.solve()
        st.write("状態", pulp.LpStatus[status])
        tmp = [xs.value() for xs in x]
        for i, a in enumerate(tmp):
            st.write(lt[i], a)
        st.write("最大価値", problem.objective.value())


if __name__ == '__main__':
    main()
