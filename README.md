# 🏛️ RCC Beam Design and Serviceability App (IS 456:2000)

A production-quality, interactive structural engineering application built with Streamlit. It performs Limit State Design for flexure and shear, and a detailed serviceability check for deflection using a Beam Finite Element Method (FEM) with variable effective stiffness ($E \cdot I_e$).

## 🚀 Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
    The application automatically loads default inputs derived from the original uploaded Excel file snippet.

## 📚 Theory and Code References

This application uses procedures consistent with **Indian Standard IS 456:2000**.

| Concept | IS 456:2000 Clause | Formula/Method Used |
| :--- | :--- | :--- |
| **Flexure (ULS)** | Cl. 38, Annex G, Cl. 26.5.1 | Limit State (0.36 fck, 0.87 fy block), Singly/Doubly check. |
| **Shear (ULS)** | Cl. 40, Table 19, Table 20 | Nominal Shear Stress ($\tau_v$), Concrete Capacity ($\tau_c$), Stirrup sizing ($V_{us}$). |
| **Empirical Deflection** | Cl. 23.2, Table 4, Fig 4 | Span-to-Effective-Depth ratio check ($L/d$). |
| **Effective Inertia ($I_e$)**| Cl. 23.3.2, Cl. 6.2.2 | **Branson's formula** interpretation: $I_e = (\frac{M_{cr}}{M_a})^3 I_g + [1 - (\frac{M_{cr}}{M_a})^3] I_{cr}$ (with $f_r = 0.7 \sqrt{f_{ck}}$). |
| **Numerical Deflection**| - | Beam Finite Element Method (Euler-Bernoulli) with $E_c \cdot I_e$. |
