import numpy as np
from typing import Dict, Any, Tuple
from scipy.optimize import fsolve

# --- IS 456:2000 Engineering Constants & Formulas ---
# Factor for ultimate moment Mu,lim/fck*b*d^2 (IS 456:2000 Table C)
KU_LIM = {250: 0.149, 415: 0.138, 500: 0.133} 
ES = 200000.0 

def get_design_constants(fy: float) -> float:
    """Returns the limit constant Ku,lim for Mu,lim."""
    if fy <= 250: return KU_LIM[250]
    elif fy <= 415: return KU_LIM[415]
    else: return KU_LIM[500]

def design_flexure(inputs: Dict[str, Any], Mu: float) -> Dict[str, Any]:
    """Performs Limit State Flexural Design (IS 456:2000, Clause 38). Mu in N-mm."""
    fck, fy, b, d = inputs['fck'], inputs['fy'], inputs['b'], inputs['d_eff']
    
    Ku_lim = get_design_constants(fy)
    Mu_lim = Ku_lim * fck * b * d**2 # N-mm
    Xu_lim_d = 0.48 if fy == 415 else 0.53 # Simplified Xu/d for Fe415/Fe250
    
    results = {'design_Mu': Mu, 'design_Mu_lim': Mu_lim, 'is_singly_reinforced': True, 'report_trace': []}
    
    results['report_trace'].append(f"M_u,lim = k_u,lim * fck * b * d^2 = {Ku_lim} * {fck} * {b} * {d}^2 = {Mu_lim:,.2f} N-mm (IS 456:2000, Cl. 38)")
    
    if Mu <= Mu_lim:
        # Singly reinforced section
        def func_xu(xu): return 0.36 * fck * b * xu * (d - 0.416 * xu) - Mu
        Xu = fsolve(func_xu, d * Xu_lim_d / 2.0)[0]
        Ast_required = (0.36 * fck * b * Xu) / (0.87 * fy)
        
        results.update({'Xu': Xu, 'Ast_required': Ast_required})
        results['report_trace'].append(f"X_u = {Xu:,.2f} mm; A_st,req = {Ast_required:,.2f} mm\u00b2 (IS 456:2000, Cl. 38.1)")
    else:
        # Doubly reinforced section
        Mu_excess = Mu - Mu_lim
        Ast2_req = Mu_excess / (0.87 * fy * (d - inputs['cover_clear'])) 
        Ast1_req = (0.36 * fck * b * Xu_lim_d * d) / (0.87 * fy)
        Ast_required = Ast1_req + Ast2_req
        
        results.update({'is_singly_reinforced': False, 'Xu': Xu_lim_d * d, 'Ast_required': Ast_required, 'Ast_compression_required': Ast2_req})
        results['report_trace'].append(f"M_u > M_u,lim. **Doubly Reinforced**. A_st,req = {Ast_required:,.2f} mm\u00b2 (IS 456:2000, Cl. G.1.1)")
        
    # Code Checks
    Ast_min = (0.85 * b * d) / fy
    Ast_max = 0.04 * b * inputs['D']
    results.update({'Ast_min': Ast_min, 'Ast_max': Ast_max})
    if results['Ast_required'] < Ast_min:
        results['Ast_required'] = Ast_min
        results['report_trace'].append(f"A_st,req adjusted to A_st,min = {Ast_min:,.2f} mm\u00b2 (Cl. 26.5.1.1)")
        
    return results

def design_shear(inputs: Dict[str, Any], Vu: float, Ast_prov: float) -> Dict[str, Any]:
    """Performs Limit State Shear Design (IS 456:2000, Clause 40). Vu in N."""
    fck, fy, b, d = inputs['fck'], inputs['fy'], inputs['b'], inputs['d_eff']
    
    tau_v = Vu / (b * d)
    pt = 100 * Ast_prov / (b * d)
    
    # Simplified Tau_c (M20, linear interpolation of Table 19)
    tau_c = 0.28
    if pt >= 0.15 and pt < 0.5: tau_c = 0.28 + (0.48 - 0.28) * (pt - 0.15) / (0.5 - 0.15)
    elif pt >= 0.5 and pt < 0.75: tau_c = 0.48 + (0.56 - 0.48) * (pt - 0.5) / (0.75 - 0.5)
    elif pt >= 0.75: tau_c = 0.56 

    tau_c_max = 2.8 # N/mm^2 for M20 (Table 20)
    
    results = {'Vu': Vu, 'tau_v': tau_v, 'tau_c': tau_c, 'tau_c_max': tau_c_max, 'report_trace': []}
    results['report_trace'].append(f"τ_v = {tau_v:,.3f} N/mm\u00b2; τ_c = {tau_c:,.3f} N/mm\u00b2 (Cl. 40.1, Table 19)")
    
    if tau_v > tau_c_max:
        results['Vs_req'] = 1e6; results['Asv_sv_req'] = 1e3
        results['report_trace'].append("!! FATAL: SECTION FAILS (τ_v > τ_c,max) !! (Cl. 40.2)")
    elif tau_v <= tau_c:
        results['Vs_req'] = 0.0
        results['Asv_sv_req'] = (0.4 * b) / (0.87 * fy) # Minimum (Cl. 26.5.1.6)
        results['report_trace'].append(f"Only Min. Shear Reinforcement required: A_sv/s_v = {results['Asv_sv_req']*1000:,.2f} mm\u00b2/m")
    else:
        results['Vc'] = tau_c * b * d
        Vs_req = Vu - results['Vc']
        Asv_sv_req = Vs_req / (0.87 * fy * d)
        results.update({'Vs_req': Vs_req, 'Asv_sv_req': Asv_sv_req})
        results['report_trace'].append(f"V_s,req = {Vs_req:,.0f} N. A_sv/s_v = {Asv_sv_req*1000:,.2f} mm\u00b2/m (Cl. 40.4)")

    return results

def calculate_effective_inertia(inputs: Dict[str, Any], M_a: float, Ast: float) -> Tuple[float, Dict[str, Any]]:
    """Calculates Effective Moment of Inertia (I_e) using Branson's (IS 456:2000, Cl. 23.3)."""
    fck, Ec, Es, b, d, D = inputs['fck'], inputs['Ec'], inputs['Es'], inputs['b'], inputs['d_eff'], inputs['D']
    
    Ig = b * D**3 / 12.0
    fr = 0.7 * np.sqrt(fck) # Modulus of Rupture (Cl. 6.2.2)
    yt = D / 2.0 
    Mcr = fr * Ig / yt
    n = Es / Ec # Modular Ratio
    
    # Cracked NA Depth (X): 0.5*b*x^2 + n*Ast*x - n*Ast*d = 0
    A, B, C = 0.5 * b, n * Ast, -n * Ast * d
    x = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    Icr = (b * x**3 / 3.0) + (n * Ast * (d - x)**2)

    Ie, formula = Ig, "I_e = I_g (Uncracked)"
    if M_a > Mcr:
        ratio_cr_a = Mcr / M_a
        Ie = (ratio_cr_a**3) * Ig + (1 - ratio_cr_a**3) * Icr # Branson's
        formula = "I_e = (M_cr/M_a)^3 * I_g + [1 - (M_cr/M_a)^3] * I_cr (Branson's - Cl. 23.3.2 interpretation)"
    
    Ie = min(Ie, Ig)
    
    trace_list = [f"I_g = {Ig:,.2f} mm\u2074", f"M_cr = {Mcr:,.2f} N-mm", f"I_cr = {Icr:,.2f} mm\u2074", f"M_a = {M_a:,.2f} N-mm", f"I_e = {Ie:,.2f} mm\u2074 ({formula})"]
    return Ie, {'Ig': Ig, 'Mcr': Mcr, 'Icr': Icr, 'Ie': Ie, 'Ec': Ec, 'I_e_trace': trace_list}

def check_empirical_deflection(inputs: Dict[str, Any], reinforcement_ratio: float) -> Dict[str, Any]:
    """Checks the empirical span/effective depth ratio (IS 456:2000, Cl. 23.2.1)."""
    L_eff, d, fy = inputs['L_eff'], inputs['d_eff'], inputs['fy']
    
    Ld_basic = 20.0 # Simply Supported, Table 4
    kt = 1.0 # Modification Factor for Tension Steel (Conservative default, Fig 4)
    kc = 1.0 # Modification Factor for Web Reinforcement (Default for beams)
    
    Ld_max = Ld_basic * kt * kc
    Ld_actual = L_eff / d
    is_safe = Ld_actual <= Ld_max
    
    trace = [f"Actual (L/d) = {Ld_actual:,.2f}", f"Allowable (L/d)_max = Basic ({Ld_basic}) * k_t ({kt}) = {Ld_max:,.2f}"]
    return {'Ld_max': Ld_max, 'Ld_actual': Ld_actual, 'is_safe': is_safe, 'report_trace': trace}

def closed_form_deflection(w_service: float, L: float, E: float, I_e: float) -> float:
    """Calculates maximum central deflection for a simply supported beam with UDL (mm)."""
    if E * I_e == 0: return np.inf
    return 5.0 * w_service * L**4 / (384.0 * E * I_e)
