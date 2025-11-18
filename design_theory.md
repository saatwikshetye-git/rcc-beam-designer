# RCC Beam Design Theory (IS 456:2000)

This document provides a concise summary of the formulas and Indian Standard (IS) 456:2000 clauses referenced in the design application for a reinforced concrete beam.

## 1. Flexural Design (Ultimate Limit State - ULS)

The design is based on the assumption of a rectangular stress block for concrete at ultimate limit state.

### 1.1 Ultimate Moment Capacity ($M_u$)
* **Neutral Axis Depth ($X_u$):** Determined by equating the total compressive force in concrete to the total tensile force in steel.
    $$0.36 f_{ck} b X_u = 0.87 f_y A_{st}$$
    Where $A_{st}$ is the required steel area.
* **Ultimate Moment of Resistance ($M_u$):** (**Cl. 38.1**)
    $$M_u = 0.87 f_y A_{st} (d - 0.416 X_u)$$
    *Alternatively, using the compression block:*
    $$M_u = 0.36 f_{ck} b X_u (d - 0.416 X_u)$$

### 1.2 Limiting Moment ($M_{u,lim}$)
* **Limiting Neutral Axis Depth ($X_{u,max}$):** The maximum depth of the neutral axis, ensured to prevent brittle concrete failure.
    $$\frac{X_{u,max}}{d} = 0.48 \quad \text{for } f_y = 415 \text{ MPa}$$
    (Value derived from **Table E**)
* **Limiting Moment ($M_{u,lim}$):** The maximum moment a singly reinforced section can resist.
    $$M_{u,lim} = 0.36 f_{ck} b X_{u,max} (d - 0.416 X_{u,max})$$
    (Factor $k_{u,lim}$ from **Table C** is used for simplification.)

### 1.3 Reinforcement Limits
* **Minimum Tension Steel ($A_{st,min}$):** (**Cl. 26.5.1.1**)
    $$\frac{A_{st,min}}{b d} = \frac{0.85}{f_y}$$
* **Maximum Tension Steel ($A_{st,max}$):** (**Cl. 26.5.1.2**)
    $$A_{st,max} = 0.04 b D$$

***

## 2. Shear Design (Ultimate Limit State - ULS)

Design is based on checking the nominal shear stress against the concrete's capacity.

* **Nominal Shear Stress ($\tau_v$):** (**Cl. 40.1**)
    $$\tau_v = \frac{V_u}{b d}$$
* **Concrete Shear Capacity ($\tau_c$):** Determined based on $f_{ck}$ and the percentage of provided tension steel ($p_t = 100 A_{st,prov} / (b d)$). (**Cl. 40.2, Table 19**)
* **Maximum Shear Stress ($\tau_{c,max}$):** Upper limit for shear stress, dictating the minimum section size. (**Cl. 40.2, Table 20**)
* **Required Stirrup Area ($A_{sv}/s_v$):** (**Cl. 40.4**)
    $$\text{If } \tau_v > \tau_c: \quad V_{us} = V_u - V_c = V_u - \tau_c b d$$   $$\frac{A_{sv}}{s_v} = \frac{V_{us}}{0.87 f_y d}$$
* **Minimum Shear Reinforcement ($A_{sv,min}$):** (**Cl. 26.5.1.6**)
    $$\frac{A_{sv}}{b s_v} \ge \frac{0.4}{0.87 f_y}$$

***

## 3. Serviceability Check: Deflection

### 3.1 Empirical Span/Depth Ratio Check
This is a quick check to satisfy deflection requirements by geometric proportion.
* **Actual Ratio:** $L_{eff} / d$
* **Allowable Ratio:** $L_{eff} / d_{max} = (\text{Basic Ratio}) \times k_t \times k_c$ (**Cl. 23.2.1, Table 4**)
    * Basic Ratio: 20 for Simply Supported beam (L < 10m).
    * $k_t$: Modification factor for tension reinforcement (**Fig 4**).
    * $k_c$: Modification factor for compression reinforcement (**Fig 5**).

### 3.2 Numerical Deflection (Effective Inertia)

The FEM solver uses the effective moment of inertia ($I_e$) to account for cracking, providing a more accurate assessment.

* **Modulus of Rupture ($f_r$):** Used to calculate the cracking moment. (**Cl. 6.2.2**)
    $$f_r = 0.7 \sqrt{f_{ck}} \quad \text{(in N/mm}^2)$$
* **Cracking Moment ($M_{cr}$):** The moment at which the section first cracks.
    $$M_{cr} = \frac{f_r I_g}{y_t}$$
    Where $I_g$ is the gross moment of inertia, and $y_t$ is the distance to the extreme tension fiber ($D/2$).
* **Effective Moment of Inertia ($I_e$):** (Branson's Formula interpretation, **Cl. 23.3.2**)
    $$\text{If } M_a > M_{cr}: \quad I_e = \left(\frac{M_{cr}}{M_a}\right)^3 I_g + \left[1 - \left(\frac{M_{cr}}{M_a}\right)^3\right] I_{cr}$$   $$\text{If } M_a \le M_{cr}: \quad I_e = I_g$$
    Where $M_a$ is the maximum moment under service loads, and $I_{cr}$ is the cracked transformed section moment of inertia.
* **Allowable Deflection:** (**Cl. 23.3.1**)
    $$\delta_{allow} = \frac{L_{eff}}{250}$$

***

## 4. Finite Element Method (FEM)

Euler-Bernoulli beam elements are used with 2 degrees of freedom (DOF) per node (vertical displacement $v$ and rotation $\theta$).

* **Element Stiffness Matrix ($k_e$):**
    $$k_e = \frac{E_c I_e}{L_e^3} \begin{bmatrix} 12 & 6 L_e & -12 & 6 L_e \\ 6 L_e & 4 L_e^2 & -6 L_e & 2 L_e^2 \\ -12 & -6 L_e & 12 & -6 L_e \\ 6 L_e & 2 L_e^2 & -6 L_e & 4 L_e^2 \end{bmatrix}$$
    Where $L_e$ is the element length, and $E_c I_e$ is the effective flexural rigidity.
* **Global System:** $\mathbf{K} \mathbf{Q} = \mathbf{F}$ is solved for nodal displacements $\mathbf{Q}$ after applying boundary conditions.
