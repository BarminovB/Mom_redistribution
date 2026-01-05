import streamlit as st
import numpy as np
import plotly.graph_objects as go
import string

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

    # Flip signs for plotting (support up, span down)
    m_el_plot = [-m for m in m_el_points]
    m_rd_plot = [-m for m in m_rd_points]

    fig = go.Figure()

    # Add elastic moment trace with hover
    fig.add_trace(go.Scatter(
        x=x_points, 
        y=m_el_plot, 
        name='Elastic moment',
        customdata=list(zip(m_el_points, x_points)),
        hovertemplate='x: %{customdata[1]:.2f} m<br>M_elastic: %{customdata[0]:.2f} kNm'
    ))

    # Add redistributed moment trace with hover
    fig.add_trace(go.Scatter(
        x=x_points, 
        y=m_rd_plot, 
        name='Moment after redistribution',
        customdata=list(zip(m_rd_points, x_points)),
        hovertemplate='x: %{customdata[1]:.2f} m<br>M_redist: %{customdata[0]:.2f} kNm'
    ))

    # Draw the beam
    max_plot = max(max(m_el_plot), max(m_rd_plot))
    beam_y = max_plot + 20  # Place beam above the moment diagram
    fig.add_shape(type="line",
                  x0=0, y0=beam_y, x1=total_length, y1=beam_y,
                  line=dict(color="black", width=4))

    # Draw supports (hinges as triangles or simple markers)
    support_names = list(string.ascii_uppercase[:n+1])
    for i in range(n+1):
        x_sup = i * L
        # Simple hinge symbol: circle below the beam
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=x_sup - 0.2, y0=beam_y - 5, x1=x_sup + 0.2, y1=beam_y - 3,
                      line_color="black", fillcolor="white")
        # Support name
        fig.add_annotation(x=x_sup, y=beam_y - 10, text=support_names[i], showarrow=False)

    fig.update_layout(
        xaxis_title='Position along the beam (m)',
        yaxis_title='Bending moment (kNm) (hogging positive up, sagging negative down)',
        title='Bending moment diagrams before and after redistribution',
        hovermode='x unified',  # Show hover for both traces at same x
        showlegend=True,
        height=600
    )

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
