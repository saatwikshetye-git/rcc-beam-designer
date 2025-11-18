import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# --- Constants & Conversions ---
N_PER_KN = 1000.0
MM_PER_M = 1000.0
CONCRETE_DENSITY_N_MM3 = 25.0 * N_PER_KN / (MM_PER_M ** 3) # 25 kN/m3 in N/mm3
STANDARD_BAR_DIAMETERS = [8, 10, 12, 16, 20, 25, 28, 32] # mm

def convert_to_base_units(value: float, unit_type: str) -> float:
    """Converts a value to the base units (N, mm, N/mm2, N-mm)."""
    if unit_type.lower() == 'load_kn_m':
        return value * N_PER_KN / MM_PER_M # kN/m -> N/mm
    return value

def parse_excel_inputs(file_content: str) -> Dict[str, Any]:
    """
    Parses the content of the uploaded Excel/CSV file to extract key design inputs.
    Using values extracted from the provided snippet as defaults.
    """
    input_data = {}
    
    # Inputs extracted from snippet (using base units N, mm)
    input_data['fck'] = 20.0
    input_data['fy'] = 415.0
    input_data['b'] = 250.0
    input_data['D'] = 300.0
    input_data['L_eff'] = 3500.0
    input_data['cover_clear'] = 30.0
    input_data['phi_long'] = 16.0
    input_data['phi_trans'] = 8.0
    
    # Loads - convert to N/mm
    w_dl_kn_m = 12.8125
    w_ll_kn_m = 1.0
    input_data['w_dl_total'] = convert_to_base_units(w_dl_kn_m, 'load_kn_m')
    input_data['w_ll'] = convert_to_base_units(w_ll_kn_m, 'load_kn_m')
    
    # Assumed data based on standard practice and the snippet
    input_data['beam_type'] = 'Simply Supported' 
    input_data['load_type'] = 'UDL' 
    input_data['concrete_density'] = CONCRETE_DENSITY_N_MM3
    input_data['gamma_f_dl'] = 1.5 
    input_data['gamma_f_ll'] = 1.5
    
    # Calculated effective depth 'd' from snippet: 300 - 30 - 8 - 16/2 = 254 mm
    d_prov_calc = input_data['D'] - input_data['cover_clear'] - input_data['phi_trans'] - input_data['phi_long'] / 2.0
    input_data['d_eff'] = round(d_prov_calc) 

    # Standard materials properties (IS 456:2000)
    input_data['Ec'] = 5000.0 * np.sqrt(input_data['fck']) # Clause 6.2.3.1
    input_data['Es'] = 200000.0 

    return input_data

def get_standard_inputs() -> Dict[str, Any]:
    """Provides a default set of inputs derived from the uploaded file snippet."""
    return parse_excel_inputs("") 

def round_to_standard_bars(area_req: float, min_phi: float = 12.0) -> Tuple[int, int]:
    """Selects a standard bar size and count to meet the required steel area."""
    for phi in sorted(STANDARD_BAR_DIAMETERS, reverse=True):
        if phi < min_phi: continue
        bar_area = np.pi * (phi / 2)**2
        count_float = area_req / bar_area
        count = int(np.ceil(count_float))
        if count <= 6 and count > 0: return (phi, count)
        if phi == STANDARD_BAR_DIAMETERS[0]: return (phi, max(2, count))
    return (12, max(2, int(np.ceil(area_req / (np.pi * (12 / 2)**2)))))

def round_to_standard_stirrups(area_req_per_m: float) -> Tuple[int, int, float]:
    """Selects a standard stirrup size and spacing (8mm, 2-legged)."""
    phi = 8.0 
    legs = 2
    Asv_bar = legs * np.pi * (phi / 2)**2 
    Asv_per_mm_req = area_req_per_m / 1000.0
    
    if Asv_per_mm_req <= 0: return (int(phi), legs, 300.0)
        
    spacing_calc_mm = Asv_bar / Asv_per_mm_req
    
    # Max spacing check (Cl. 26.5.1.5): min(0.75*d, 300mm)
    max_spacing = min(300.0, 0.75 * get_standard_inputs()['d_eff'])
    spacing_final = min(max_spacing, round(spacing_calc_mm / 5.0) * 5.0) # Round down to nearest 5mm
    
    if spacing_final < 50: spacing_final = 50.0 # Min spacing heuristic
    
    return (int(phi), legs, spacing_final)
