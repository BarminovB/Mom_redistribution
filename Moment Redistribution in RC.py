import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Beam Moment Redistribution Calculator")

    n_spans = st.number_input("Enter number of spans:", min_value=1, value=2, step=1)
    q = st.number_input("Enter linear load q (kN/m):", value=32.0)
    L = st.number_input("Enter span length L (m):", value=6.0)
    delta = st.number_input("Enter redistribution coefficient δ:", value=0.7)

    if n_spans < 2:
        st.write("For one span, redistribution does not apply.")
        return

    n = n_spans
    k = n - 1
    A = np.zeros((k, k))
    b = np.full(k, -q * L**2 / 2)

    for i in range(k):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = 1
        if i < k-1:
            A[i, i+1] = 1

    M_internal = np.linalg.solve(A, b)
    M_supports_el = [0.0] + list(M_internal) + [0.0]
    M_supports_rd = [0.0] + [delta * m for m in M_internal] + [0.0]

    st.write("Elastic support moments (kNm):", [round(m, 2) for m in M_supports_el])
    st.write("Redistributed support moments (kNm):", [round(m, 2) for m in M_supports_rd])

    for span in range(n):
        def m_el(x_loc):
            return M_supports_el[span] * (1 - x_loc / L) + M_supports_el[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2

        x_locs = np.linspace(0, L, 100)
        m_els = [m_el(x) for x in x_locs]
        max_m_el = max(m_els)
        st.write(f"Span {span + 1}, max span moment elastic: {max_m_el:.2f} kNm")

        def m_rd(x_loc):
            return M_supports_rd[span] * (1 - x_loc / L) + M_supports_rd[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2

        m_rds = [m_rd(x) for x in x_locs]
        max_m_rd = max(m_rds)
        st.write(f"Span {span + 1}, max span moment after redistribution: {max_m_rd:.2f} kNm")

    total_length = n * L
    x_points = np.linspace(0, total_length, 1000)
    m_el_points = []
    m_rd_points = []

    for x in x_points:
        span = min(int(x // L), n - 1)
        x_loc = x - span * L
        m_el = M_supports_el[span] * (1 - x_loc / L) + M_supports_el[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
        m_el_points.append(m_el)
        m_rd = M_supports_rd[span] * (1 - x_loc / L) + M_supports_rd[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
        m_rd_points.append(m_rd)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_points, m_el_points, label='Elastic moment')
    ax.plot(x_points, m_rd_points, label='Moment after redistribution')
    ax.set_xlabel('Position along the beam (m)')
    ax.set_ylabel('Bending moment (kNm)')
    ax.set_title('Bending moment diagrams before and after redistribution')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
