import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple
import io
import fem # Ensure fem is imported for post-processing

def generate_plot_bytes(x_data: List[float], y_data: List[float], title: str, xlabel: str, ylabel: str, color: str = 'b', is_moment_shear: bool = False, line_style: str = '-') -> bytes:
    """Generates a high-quality Matplotlib plot and returns it as a PNG byte object."""
    
    # Use a large enough figure size and high DPI for good PDF quality
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150) 
    
    x_m = np.array(x_data) / 1000.0
    y_plot = np.array(y_data)
    
    # Scaling for display units (kN and kN-m)
    if 'Moment' in title:
        y_plot = y_plot / 1e6 # N-mm to kN-m
        if is_moment_shear: ax.invert_yaxis() 
    elif 'Shear' in title:
        y_plot = y_plot / 1000.0 # N to kN

    ax.plot(x_m, y_plot, color=color, linestyle=line_style)
    
    # Fill only for SFD/BMD
    if is_moment_shear:
        ax.fill_between(x_m, 0, y_plot, alpha=0.2, color=color)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    buf = io.BytesIO()
    # bbox_inches='tight' prevents labels from being cut off
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def plot_results(L_eff: float, w_factored: float, fem_data: Dict[str, Any], I_e_data: Dict[str, Any]) -> Dict[str, bytes]:
    """Generates all required plots."""
    
    # Calculate ULS diagrams (M and V)
    x_plot, M_plot, V_plot = fem.fem_post_process_m_v(w_factored, L_eff)
    
    # Get SLS data
    x_nodes, v_deflection = fem_data.get('nodes', x_plot), fem_data.get('deflection', np.zeros_like(x_plot))
    
    # Calculate Curvature (using service moment and effective stiffness)
    w_service = w_factored / 1.5 # Assuming factor of 1.5
    M_service = fem.fem_post_process_m_v(w_service, L_eff)[1]
    E_c = I_e_data.get('Ec', 1.0)
    I_e = I_e_data.get('Ie', 1e12) # Use a large default Ie to prevent division by zero
    curvature = M_service / (E_c * I_e) # 1/mm
    
    plots_png = {
        '1. Shear Diagram (V_u)': generate_plot_bytes(x_plot, V_plot, "Shear Force Diagram (Factored)", "Span (m)", "Shear Force (kN)", '#007bff', True),
        '2. Bending Moment Diagram (M_u)': generate_plot_bytes(x_plot, M_plot, "Bending Moment Diagram (Factored)", "Span (m)", "Bending Moment (kN-m)", '#dc3545', True),
        # Deflection array is negative for downward deflection, plot positive downward
        '3. Deflection Curve (Service Load)': generate_plot_bytes(x_nodes, -v_deflection, "Deflection Curve (Service Load, FEM)", "Span (m)", "Deflection (mm)", '#ffc107', False),
        '4. Curvature Diagram (Service Load)': generate_plot_bytes(x_plot, curvature * 1000, "Curvature Diagram (Service Load)", "Span (m)", "Curvature (1/m)", '#6f42c1', False)
    }

    return plots_png
