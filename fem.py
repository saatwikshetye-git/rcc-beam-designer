import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, List, Tuple

def element_stiffness(E: float, I: float, L_e: float) -> np.ndarray:
    """4x4 stiffness matrix for a 2-node beam element (DOF: [v1, theta1, v2, theta2])."""
    if L_e == 0: return np.zeros((4, 4))
    EI_L3 = E * I / (L_e**3)
    EI_L2 = E * I / (L_e**2)
    EI_L = E * I / L_e
    k = np.array([
        [12*EI_L3, 6*EI_L2, -12*EI_L3, 6*EI_L2],
        [6*EI_L2, 4*EI_L, -6*EI_L2, 2*EI_L],
        [-12*EI_L3, -6*EI_L2, 12*EI_L3, -6*EI_L2],
        [6*EI_L2, 2*EI_L, -6*EI_L2, 4*EI_L]
    ])
    return k

def fem_solver(inputs: Dict[str, Any], I_e: float, E: float, n_elements: int = 20) -> Dict[str, Any]:
    """FEM solver for Simply Supported Beam with UDL."""
    L = inputs['L_eff']
    w_service = inputs['w_dl_total'] + inputs['w_ll'] # N/mm
    
    L_e = L / n_elements
    n_dof = 2 * (n_elements + 1)
    K = lil_matrix((n_dof, n_dof), dtype=float)
    F = np.zeros(n_dof)
    k_e = element_stiffness(E, I_e, L_e)
    
    for i in range(n_elements):
        dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        
        # Assembly of K
        for r_local, r_global in enumerate(dofs):
            for c_local, c_global in enumerate(dofs):
                K[r_global, c_global] += k_e[r_local, c_local]
                
        # Load contribution from UDL: [wL_e/2, wL_e^2/12, wL_e/2, -wL_e^2/12]
        f_e = [w_service * L_e / 2, w_service * L_e**2 / 12, w_service * L_e / 2, -w_service * L_e**2 / 12]
        for r_local, r_global in enumerate(dofs):
            F[r_global] += f_e[r_local]

    K_csr = K.tocsr()
    
    # Boundary Conditions (SS): v1=0 (DOF 0), v_N+1=0 (DOF 2*n_elements)
    constrained_dofs = [0, 2 * n_elements]
    free_dofs = [i for i in range(n_dof) if i not in constrained_dofs]
    
    K_ff = K_csr[free_dofs, :][:, free_dofs]
    F_f = F[free_dofs]
    
    # Solve for free displacements Q_f
    Q_f = spsolve(K_ff, F_f)
    
    Q = np.zeros(n_dof)
    for i, dof in enumerate(free_dofs):
        Q[dof] = Q_f[i]
        
    nodes = np.linspace(0, L, n_elements + 1)
    deflection_v = Q[::2]
    slope_theta = Q[1::2]
    
    return {
        'nodes': nodes, 
        'deflection': deflection_v, # v (mm)
        'slope': slope_theta,       # theta (rad)
        'max_deflection': -np.min(deflection_v),
        'n_elements': n_elements
    }

def fem_post_process_m_v(w_factored: float, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates SFD and BMD using closed-form for SS+UDL (N-mm, N)."""
    x = np.linspace(0, L, 100)
    R = w_factored * L / 2.0
    V = R - w_factored * x
    M = R * x - w_factored * x**2 / 2.0
    return x, M, V
