import numpy as np
import matplotlib.pyplot as plt

def main():
    n_spans = int(input("Enter number of spans: "))
    q = float(input("Enter linear load q (kN/m): "))
    L = float(input("Enter span length L (m): "))
    delta = float(input("Enter redistribution coefficient δ: "))

    if n_spans < 2:
        print("For one span, redistribution does not apply.")
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

    print("Elastic support moments (kNm):", [round(m, 2) for m in M_supports_el])
    print("Redistributed support moments (kNm):", [round(m, 2) for m in M_supports_rd])

    for span in range(n):
        def m_el(x_loc):
            return M_supports_el[span] * (1 - x_loc / L) + M_supports_el[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2

        x_locs = np.linspace(0, L, 100)
        m_els = [m_el(x) for x in x_locs]
        max_m_el = max(m_els)
        print(f"Span {span + 1}, max span moment elastic: {max_m_el:.2f} kNm")

        def m_rd(x_loc):
            return M_supports_rd[span] * (1 - x_loc / L) + M_supports_rd[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2

        m_rds = [m_rd(x) for x in x_locs]
        max_m_rd = max(m_rds)
        print(f"Span {span + 1}, max span moment after redistribution: {max_m_rd:.2f} kNm")

    total_length = n * L
    x_points = np.linspace(0, total_length, 1000)
    m_el_points = []
    m_rd_points = []

    for x in x_points:
        span = int(x // L)
        x_loc = x % L
        m_el = M_supports_el[span] * (1 - x_loc / L) + M_supports_el[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
        m_el_points.append(m_el)
        m_rd = M_supports_rd[span] * (1 - x_loc / L) + M_supports_rd[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
        m_rd_points.append(m_rd)

    plt.figure(figsize=(12, 6))
    plt.plot(x_points, m_el_points, label='Elastic moment')
    plt.plot(x_points, m_rd_points, label='Moment after redistribution')
    plt.xlabel('Position along the beam (m)')
    plt.ylabel('Bending moment (kNm)')
    plt.title('Bending moment diagrams before and after redistribution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()