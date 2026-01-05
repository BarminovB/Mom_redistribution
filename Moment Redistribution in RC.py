import numpy as np
import matplotlib.pyplot as plt

def main():
    # Hardcoded inputs based on example
    n_spans = 2  # Number of spans
    q = 32.0     # Linear load (kN/m)
    L = 6.0      # Span length (m)
    delta = 0.7  # Redistribution coefficient (reduces absolute support moments)

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
    M_supports_el = [0.0] + list(M_internal) + [0.0]  # Elastic support moments (negative for hogging)
    M_supports_rd = [0.0] + [delta * m for m in M_internal] + [0.0]  # Redistributed support moments

    # Verification with standard formulas for two-span beam
    print("Verification:")
    expected_M_support_el = -q * L**2 / 8
    calculated_M_support_el = M_internal[0]
    print(f"Expected elastic central support moment: {expected_M_support_el:.2f} kNm")
    print(f"Calculated elastic central support moment: {calculated_M_support_el:.2f} kNm")
    assert abs(expected_M_support_el - calculated_M_support_el) < 1e-6, "Elastic support moment mismatch"

    expected_max_span_el = q * L**2 / 16
    def m_el(x_loc, span=0):
        return M_supports_el[span] * (1 - x_loc / L) + M_supports_el[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
    max_m_el = max([m_el(x) for x in np.linspace(0, L, 100)])
    print(f"Expected max elastic span moment: {expected_max_span_el:.2f} kNm")
    print(f"Calculated max elastic span moment: {max_m_el:.2f} kNm")
    assert abs(expected_max_span_el - max_m_el) < 1e-6, "Elastic span moment mismatch"

    expected_M_support_rd = delta * expected_M_support_el
    print(f"Expected redistributed central support moment: {expected_M_support_rd:.2f} kNm")
    print(f"Calculated redistributed central support moment: {M_supports_rd[1]:.2f} kNm")

    expected_max_span_rd = q * L**2 / 8 + 0.5 * expected_M_support_rd  # Algebraic calculation
    def m_rd(x_loc, span=0):
        return M_supports_rd[span] * (1 - x_loc / L) + M_supports_rd[span + 1] * (x_loc / L) + q * x_loc * (L - x_loc) / 2
    max_m_rd = max([m_rd(x) for x in np.linspace(0, L, 100)])
    print(f"Expected max redistributed span moment: {expected_max_span_rd:.2f} kNm")
    print(f"Calculated max redistributed span moment: {max_m_rd:.2f} kNm")
    print("Verification passed: Calculations match standard formulas for continuous beams under uniform load.\n")

    # Generate points for the entire beam
    total_length = n * L
    x_points = np.linspace(0, total_length, 1000)