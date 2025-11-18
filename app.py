import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
from typing import Dict, Any, Optional
import time
import os
import io

# Import the existing backend modules (DO NOT CHANGE THESE)
try:
    import io_utils
    import design
    import fem
    import plots
    import report
except ImportError as e:
    st.error(f"FATAL ERROR: Could not import backend module. Ensure files are present: {e}")
    st.stop()

# --- Configuration & Constants ---
APP_TITLE = "RCC Beam Designer Pro"
APP_SUBTITLE = "Limit State Design & Serviceability (IS 456:2000)"
APP_VERSION = "v1.1.0"
APP_AUTHOR = "by Saatwik Shetye"
ACCENT_COLOR = "#1d3557"  # Dark blue for design elements
L_DEFLECTION_LIMIT_RATIO = 250.0

# --- Initialize Streamlit page config early ---
st.set_page_config(layout="wide", page_title=APP_TITLE, initial_sidebar_state="expanded")

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = "1. Inputs"

if 'inputs' not in st.session_state:
    # Expect io_utils.get_standard_inputs() to provide a dictionary with standard keys
    st.session_state.inputs = io_utils.get_standard_inputs()
if 'design_results' not in st.session_state:
    st.session_state.design_results = None
if 'fem_results' not in st.session_state:
    st.session_state.fem_results = None
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = '(B) Detailed Limit-State'
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = None

# --- UI Helpers ---

def inject_css():
    """Injects custom CSS for visual polish and includes JavaScript to reset scroll position."""
    st.markdown(f"""
    <style>
        /* Page container padding */
        .reportview-container .main .block-container {{
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }}

        .metric-label p, .metric-value p {{
            font-weight: bold;
        }}

        .badge-pass {{
            background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold;
        }}
        .badge-fail {{
            background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold;
        }}
        .badge-warn {{
            background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold;
        }}

        /* Prevent the browser from trying to maintain scroll position on reruns */
        .main, .stApp {{
            scroll-behavior: auto !important;
        }}
    </style>
    <script>
        // Reset scroll to top on load / rerun
        setTimeout(() => {{
            window.scrollTo({{top: 0, behavior: 'auto'}});
        }}, 0);
    </script>
    """, unsafe_allow_html=True)

def update_inputs_from_ui(current_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Recalculates dependent geometry/material properties after UI changes."""
    updated_inputs = current_inputs.copy()

    # Calculate effective depth d_eff (rounded)
    # NOTE: this uses fields expected to be in inputs: D, cover_clear, phi_trans, phi_long
    try:
        d_prov_calc = updated_inputs['D'] - updated_inputs['cover_clear'] - updated_inputs['phi_trans'] - updated_inputs['phi_long'] / 2.0
        updated_inputs['d_eff'] = float(round(d_prov_calc, 0))
    except Exception:
        # Fallback: set d_eff to D - cover_clear - 50 if keys missing
        updated_inputs['d_eff'] = updated_inputs.get('D', 300.0) - updated_inputs.get('cover_clear', 25.0) - 50.0

    # Concrete modulus of elasticity (Ec) in N/mm^2: Ec = 5000 * sqrt(fck)
    try:
        updated_inputs['Ec'] = 5000.0 * np.sqrt(updated_inputs['fck'])
    except Exception:
        updated_inputs['Ec'] = 5000.0 * np.sqrt(updated_inputs.get('fck', 20.0))

    return updated_inputs

def create_cross_section_plot(inputs: Dict[str, Any], reinforcement_details: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Generates an interactive 2D plot of the beam cross-section."""
    b = float(inputs.get('b', 250.0))
    D = float(inputs.get('D', 300.0))
    cover_clear = float(inputs.get('cover_clear', 25.0))
    phi_trans = float(inputs.get('phi_trans', 8.0))
    phi_long_input = float(inputs.get('phi_long', 16.0))

    # Default reinforcement if not designed yet
    if reinforcement_details:
        phi_long = reinforcement_details.get('phi_long', phi_long_input)
        count_long = reinforcement_details.get('count_long', 3)
    else:
        phi_long = phi_long_input
        count_long = 3

    # Effective vertical position of bar centers (from bottom)
    d_eff_calc = D - cover_clear - phi_trans - phi_long / 2.0
    bar_radius = phi_long / 2.0

    # Simple placement heuristic for 1..6 bars in single layer
    if count_long > 1:
        # available width for bars between clear covers and stirrup thicknesss
        available_width = b - 2 * cover_clear - 2 * phi_trans
        # keep each bar diameter phi_long and distribute evenly
        spacing = available_width - phi_long
        if spacing < 0 or count_long == 1:
            spacing = 0.0
        else:
            spacing = spacing / (count_long - 1)
    else:
        spacing = 0.0

    bar_x = []
    start_x = cover_clear + phi_trans + bar_radius
    for i in range(count_long):
        bar_x.append(start_x + i * spacing)

    bar_y = [d_eff_calc] * count_long

    fig = go.Figure()

    # Concrete (Rectangle)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=b, y1=D,
        line=dict(color=ACCENT_COLOR, width=2),
        fillcolor="rgba(192, 192, 192, 0.5)",
    )

    # Stirrups (inner rectangle indicating clear cover zone)
    stirrup_x0 = cover_clear
    stirrup_y0 = cover_clear
    stirrup_x1 = b - cover_clear
    stirrup_y1 = D - cover_clear
    fig.add_shape(
        type="rect",
        x0=stirrup_x0, y0=stirrup_y0, x1=stirrup_x1, y1=stirrup_y1,
        line=dict(color='gray', width=1, dash='dash'),
        fillcolor="rgba(0,0,0,0)",
    )

    # Longitudinal Bars (Scatter/Markers)
    fig.add_trace(go.Scatter(
        x=bar_x,
        y=bar_y,
        mode='markers',
        marker=dict(size=phi_long, color='#a00000', symbol='circle'),
        name=f'{count_long} x φ{phi_long}'
    ))

    fig.update_layout(
        title='Beam Cross-Section Preview (mm)',
        xaxis_title="Width (mm)",
        yaxis_title="Depth (mm)",
        xaxis=dict(range=[0, b], constrain='range'),
        yaxis=dict(range=[0, D], scaleanchor="x"),
        showlegend=True,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_diagram_plot(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, unit: str, color: str, invert_y: bool = False) -> go.Figure:
    """Generates an interactive Plotly diagram for V, M, or Deflection."""
    # Convert x (mm) to meters for plotting along beam axis
    x_m = np.array(x) / 1000.0

    # Scale y according to unit
    if unit == 'kN':
        y_plot = np.array(y) / 1000.0
    elif unit == 'kN-m':
        y_plot = np.array(y) / 1e6
    else:
        # mm, 1/m etc keep as-is (or scaled before passing)
        y_plot = np.array(y)

    # Convert hex color to rgba for translucent fill
    try:
        rgb_tuple = pcolors.hex_to_rgb(color)
        rgb_str = f'rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, 0.2)'
    except Exception:
        rgb_str = "rgba(0,0,0,0.1)"

    fig = go.Figure()
    # Main curve with filled area
    fig.add_trace(go.Scatter(
        x=x_m, y=y_plot, mode='lines', line=dict(color=color, width=3),
        fill='tozeroy', fillcolor=rgb_str, name=title
    ))

    # Highlight max absolute value point
    try:
        max_abs_idx = int(np.nanargmax(np.abs(y_plot)))
        fig.add_trace(go.Scatter(
            x=[x_m[max_abs_idx]], y=[y_plot[max_abs_idx]], mode='markers',
            marker=dict(size=10, color='red'), name=f'Max |{ylabel}|'
        ))
    except Exception:
        pass

    fig.update_layout(
        title=f'<b>{title}</b>',
        xaxis_title="Position along Beam (m)",
        yaxis_title=f"{ylabel} ({unit})",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )

    if invert_y:
        fig.update_yaxes(autorange="reversed")

    return fig

# --- Core Computation Function ---

def run_full_analysis():
    """Manages the step-by-step computation and updates session state."""
    st.session_state.design_results = None
    st.session_state.fem_results = None
    st.session_state.plot_data = None
    inputs = st.session_state.inputs

    # Steps and progress UI placeholders
    progress_steps = [
        "1. Calculating Mu/Vu",
        "2. Flexure Design (ULS)",
        "3. Shear Design (ULS)",
        "4. Effective Inertia (Ie)",
        "5. FEM Analysis (SLS)"
    ]
    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        # Step 1: Preliminary Calcs (Mu, Vu)
        progress_text.info(progress_steps[0])
        gamma_f = float(inputs.get('gamma_f_dl', 1.5))
        # w_dl_total and w_ll expected in N/mm (or base units as per io_utils)
        w_factored = gamma_f * (inputs.get('w_dl_total', 0.0) + inputs.get('w_ll', 0.0))  # N/mm
        L = float(inputs.get('L_eff', 3500.0))  # mm
        Mu = w_factored * L**2 / 8.0  # N-mm
        Vu = w_factored * L / 2.0  # N
        progress_bar.progress(10)

        # Step 2: Flexural Design
        progress_text.info(progress_steps[1])
        flexure_res = design.design_flexure(inputs, Mu)
        Ast_req = flexure_res.get('Ast_required', 0.0)
        # io_utils.round_to_standard_bars expected to return (phi_long, count_long)
        phi_long, count_long = io_utils.round_to_standard_bars(Ast_req, min_phi=int(inputs.get('phi_long', 12)))
        Ast_prov = count_long * np.pi * (phi_long / 2.0) ** 2

        flexure_res['Ast_prov'] = Ast_prov
        reinforcement_details = {
            'phi_long': phi_long,
            'count_long': count_long,
            'Ast_prov': Ast_prov,
            'Vu': Vu,
            'Mu': Mu
        }
        progress_bar.progress(30)

        # Step 3: Shear Design
        progress_text.info(progress_steps[2])
        shear_res = design.design_shear(inputs, Vu, Ast_prov)
        # design.design_shear is expected to return values where Asv_sv_req is in appropriate units
        Asv_sv_req = shear_res.get('Asv_sv_req', 0.0) * 1000.0  # convert if necessary
        phi_trans, legs, spacing_trans = io_utils.round_to_standard_stirrups(Asv_sv_req)
        reinforcement_details.update({'phi_trans': phi_trans, 'spacing_trans': spacing_trans})
        progress_bar.progress(45)

        # Step 4: Effective Inertia
        progress_text.info(progress_steps[3])
        w_service = inputs.get('w_dl_total', 0.0) + inputs.get('w_ll', 0.0)
        M_a = w_service * L**2 / 8.0
        Ie, Ie_data = design.calculate_effective_inertia(inputs, M_a, Ast_prov)
        reinforcement_ratio = 0.0
        try:
            reinforcement_ratio = Ast_prov * 100.0 / (inputs['b'] * inputs['d_eff'])
        except Exception:
            reinforcement_ratio = 0.0
        empirical_res = design.check_empirical_deflection(inputs, reinforcement_ratio)
        progress_bar.progress(65)

        # Step 5: FEM Analysis (if Detailed)
        progress_text.info(progress_steps[4])
        if st.session_state.analysis_mode == '(B) Detailed Limit-State':
            fem_res_refined = fem.fem_solver(inputs, Ie, inputs.get('Ec', 0.0), n_elements=40)
            fem_res_coarse = fem.fem_solver(inputs, Ie, inputs.get('Ec', 0.0), n_elements=20)
            max_def_ref = fem_res_refined.get('max_deflection', 0.0)
            max_def_coarse = fem_res_coarse.get('max_deflection', 0.0)
            delta_percent = np.nan
            try:
                if max_def_ref != 0:
                    delta_percent = abs(max_def_ref - max_def_coarse) / max_def_ref * 100.0
            except Exception:
                delta_percent = np.nan

            fem_results_out = {
                'fem_results': fem_res_refined,
                'I_e_data': Ie_data,
                'convergence_check': {'delta_percent': delta_percent}
            }
        else:
            # Quick mode: closed-form
            max_def_closed = design.closed_form_deflection(w_service, L, inputs.get('Es', inputs.get('Ec', 0.0)), Ie)
            x_plot, M_plot_dummy, V_plot_dummy = fem.fem_post_process_m_v(w_factored, L)
            fem_results_out = {
                'fem_results': {'max_deflection': max_def_closed, 'nodes': x_plot, 'deflection': np.zeros_like(x_plot)},
                'I_e_data': Ie_data,
                'convergence_check': {'delta_percent': np.nan}
            }
        progress_bar.progress(85)

        # Finalization
        design_results_out = {
            'flexure': flexure_res,
            'shear': shear_res,
            'reinforcement': reinforcement_details,
            'deflection': {'empirical': empirical_res}
        }

        st.session_state.design_results = design_results_out
        st.session_state.fem_results = fem_results_out

        # Pre-generate plot data for Plotly
        x_plot, M_plot, V_plot = fem.fem_post_process_m_v(w_factored, L)
        # For service moment (use un-factored w for SLS)
        M_service = fem.fem_post_process_m_v(w_service, L)[1]
        Ec = inputs.get('Ec', 5000.0 * np.sqrt(inputs.get('fck', 20)))
        curvature = M_service / (Ec * Ie) if Ie != 0 else np.zeros_like(M_service)

        st.session_state.plot_data = {
            'x_plot': x_plot,
            'M_plot': M_plot,
            'V_plot': V_plot,
            'x_nodes': fem_results_out['fem_results'].get('nodes', x_plot),
            'v_deflection': fem_results_out['fem_results'].get('deflection', np.zeros_like(x_plot)),
            'curvature': curvature,
            'L_eff': L
        }

        progress_bar.progress(100)
        progress_text.success("Analysis Complete! 🎉")
        time.sleep(0.3)
    except Exception as e:
        progress_bar.empty()
        progress_text.error(f"Analysis failed. Check inputs. Error: {e}")
        st.session_state.design_results = None
        st.session_state.fem_results = None
        st.session_state.plot_data = None

# --- Pages ---

def page_inputs():
    """Sidebar page for defining all design inputs."""
    st.header("1. 🏗️ Geometry")
    inputs = st.session_state.inputs

    col_b, col_D = st.columns(2)
    inputs['b'] = col_b.number_input("Width $b$ (mm)", value=float(inputs.get('b', 250.0)), min_value=100.0, step=10.0, key='in_b')
    inputs['D'] = col_D.number_input("Depth $D$ (mm)", value=float(inputs.get('D', 300.0)), min_value=100.0, step=10.0, key='in_D')

    # L_eff input: stored in inputs in mm, display in meters
    inputs['L_eff'] = float(st.number_input("Span $L_{eff}$ (m)", value=float(inputs.get('L_eff', 3500.0)) / 1000.0, min_value=0.5, step=0.1, key='in_L') * 1000.0)

    with st.expander("Cover & Bar Properties (Fixed in Backend)"):
        st.caption(f"Clear Cover: {inputs.get('cover_clear', 25.0)} mm. Stirrup $\\phi$: {inputs.get('phi_trans', 8.0)} mm. Longitudinal $\\phi$: {inputs.get('phi_long', 16.0)} mm.")

    st.header("2. ⚖️ Loads")
    col_dl, col_ll = st.columns(2)
    # Display in kN/m for user; internal base unit conversions handled by io_utils
    w_dl_kn_m = float(inputs.get('w_dl_total', 0.0)) * 1000.0
    w_ll_kn_m = float(inputs.get('w_ll', 0.0)) * 1000.0
    w_dl_kn_m = col_dl.number_input("Dead Load $w_{DL}$ (kN/m)", value=w_dl_kn_m, min_value=0.0, step=0.1, key='in_dl')
    w_ll_kn_m = col_ll.number_input("Live Load $w_{LL}$ (kN/m)", value=w_ll_kn_m, min_value=0.0, step=0.1, key='in_ll')

    # Convert back to base units using backend helper
    try:
        inputs['w_dl_total'] = io_utils.convert_to_base_units(w_dl_kn_m, 'load_kn_m')
    except Exception:
        # If conversion helper not available or fails, assume user provided kN/m -> convert to N/mm
        inputs['w_dl_total'] = (w_dl_kn_m * 1000.0) / 1000.0  # fallback

    try:
        inputs['w_ll'] = io_utils.convert_to_base_units(w_ll_kn_m, 'load_kn_m')
    except Exception:
        inputs['w_ll'] = (w_ll_kn_m * 1000.0) / 1000.0

    st.header("3. 🧱 Materials")
    col_fck, col_fy = st.columns(2)
    inputs['fck'] = col_fck.number_input("Concrete $f_{ck}$ (MPa)", value=float(inputs.get('fck', 20.0)), min_value=10.0, step=5.0, key='in_fck')
    inputs['fy'] = col_fy.number_input("Steel $f_{y}$ (MPa)", value=float(inputs.get('fy', 415.0)), min_value=250.0, step=5.0, key='in_fy')

    # Update calculated properties and save state
    st.session_state.inputs = update_inputs_from_ui(inputs)

    st.divider()
    st.info(f"Calculated Effective Depth $d$: **{st.session_state.inputs.get('d_eff', 0):.0f} mm**")

    # Show cross-section preview (if any design results exist, pass reinforcement)
    reinforcement_data = st.session_state.design_results.get('reinforcement') if st.session_state.design_results else None
    st.plotly_chart(create_cross_section_plot(st.session_state.inputs, reinforcement_data), use_container_width=True)

def page_results():
    """Main page for displaying design results."""
    st.header("Design Results & Summary")

    if st.session_state.design_results is None:
        st.warning("Please run the analysis first.")
        return

    design_res = st.session_state.design_results
    fem_res = st.session_state.fem_results

    reinforcement = design_res.get('reinforcement', {})
    flexure = design_res.get('flexure', {})
    shear = design_res.get('shear', {})

    # --- Metrics Section (The Status Lights) ---
    st.subheader("Key Design Checks")
    col_m, col_v, col_ast, col_shear = st.columns(4)

    # Safely read fields for display
    Mu_display = reinforcement.get('Mu', 0.0) / 1e6
    Vu_display = reinforcement.get('Vu', 0.0) / 1000.0
    Ast_required = flexure.get('Ast_required', 0.0)

    col_m.metric("Max Moment $M_u$", f"{Mu_display:,.2f} kN-m")
    col_v.metric("Max Shear $V_u$", f"{Vu_display:,.2f} kN")
    col_ast.metric("Required $A_{st}$", f"{Ast_required:,.1f} mm\u00b2")

    # Shear check badge
    shear_check = "PASS"
    try:
        shear_check = "PASS" if shear.get('tau_v', 0.0) <= shear.get('tau_c_max', np.inf) else "FAIL (Section too small!)"
    except Exception:
        shear_check = "UNKNOWN"

    shear_badge_class = "pass" if "PASS" in shear_check else "fail"
    shear_badge_html = f"<span class='badge-{shear_badge_class}'> Shear {shear_check}</span>"
    col_shear.markdown(f"**Shear Check** {shear_badge_html}", unsafe_allow_html=True)

    st.divider()

    # --- ULS Design Details (Flexure & Shear) ---
    st.subheader("Ultimate Limit State (ULS) Details")
    col_flex, col_shear_det = st.columns(2)

    with col_flex:
        st.markdown("#### 🔨 Flexural Design")
        try:
            st.markdown(f"**$M_{{u,lim}}$**: {flexure.get('design_Mu_lim', 0.0) / 1e6:,.2f} kN-m")
            st.markdown(f"**Section Type**: {'Singly Reinforced' if flexure.get('is_singly_reinforced', True) else 'Doubly Reinforced'}")
            st.success(f"**Provided $\\text{{A}}_{{\\text{{st,prov}}}}$**: {reinforcement.get('count_long', '?')} x $\\phi${reinforcement.get('phi_long', '?')} ({reinforcement.get('Ast_prov', 0.0):,.2f} $\\text{{mm}}^2$)")
        except Exception:
            st.info("Flexure results incomplete.")

        with st.expander("Flexure Calculation Trace"):
            for line in flexure.get('report_trace', []):
                st.code(line)

    with col_shear_det:
        st.markdown("#### 🔪 Shear Design")
        try:
            st.markdown(f"**Nominal $\\tau_v$**: {shear.get('tau_v', 0.0):.3f} N/mm\u00b2")
            st.markdown(f"**Concrete $\\tau_c$**: {shear.get('tau_c', 0.0):.3f} N/mm\u00b2")
        except Exception:
            st.info("Shear results incomplete.")

        if shear.get('Vs_req', 0.0) > 0:
            st.warning(f"**Stirrups Required** $\\text{{V}}_{{\\text{{us}}}}$: {shear.get('Vs_req', 0.0):.0f} N")
        else:
            st.success("Only Minimum Stirrups Required.")

        st.warning(f"**Provided Stirrups**: $\\phi${reinforcement.get('phi_trans', '?')} @ {reinforcement.get('spacing_trans', 0.0):.0f} mm c/c")

        with st.expander("Shear Calculation Trace"):
            for line in shear.get('report_trace', []):
                st.code(line)

    st.divider()

    # --- SLS Design Details (Deflection) ---
    st.subheader("Serviceability Limit State (SLS) Details")
    col_sls, col_fem = st.columns(2)

    with col_sls:
        st.markdown("#### Empirical Check (L/d)")
        empirical = design_res.get('deflection', {}).get('empirical', {})
        Ld_actual = empirical.get('Ld_actual', np.inf)
        Ld_max = empirical.get('Ld_max', L_DEFLECTION_LIMIT_RATIO)
        deflection_pass = Ld_actual <= Ld_max

        Ld_badge_class = 'pass' if deflection_pass else 'fail'
        Ld_badge_html = f"<span class='badge-{Ld_badge_class}'> L/d {'PASS' if deflection_pass else 'FAIL'}</span>"
        st.markdown(f"**Result** {Ld_badge_html}", unsafe_allow_html=True)
        st.info(f"Actual $L/d$: {Ld_actual:.2f} | Max $L/d$: {Ld_max:.2f}")

    with col_fem:
        st.markdown("#### Numerical Check (FEM)")
        max_def_fem = fem_res.get('fem_results', {}).get('max_deflection', np.inf)
        L_eff = st.session_state.inputs.get('L_eff', 3500.0)
        allow_def = L_eff / L_DEFLECTION_LIMIT_RATIO
        fem_pass = (max_def_fem <= allow_def) if np.isfinite(max_def_fem) else False

        fem_badge_class = 'pass' if fem_pass else 'fail'
        fem_badge_html = f"<span class='badge-{fem_badge_class}'> FEM Deflection {'PASS' if fem_pass else 'FAIL'}</span>"
        st.markdown(f"**Result** {fem_badge_html}", unsafe_allow_html=True)

        if np.isfinite(max_def_fem):
            st.info(f"Max $\\delta_{{max}}$: {max_def_fem:.3f} mm | Allow $\\delta_{{allow}}$ (L/250): {allow_def:.3f} mm")
        else:
            st.info("Max deflection data not available.")

        if st.session_state.analysis_mode == '(B) Detailed Limit-State':
            delta_percent = fem_res.get('convergence_check', {}).get('delta_percent', np.nan)
            caption_text = f"Convergence Check Δ: {delta_percent:.2f}%"
            if not np.isnan(delta_percent):
                st.caption(f"{caption_text} {'(OK)' if delta_percent < 2.0 else '(WARN)'}")
            else:
                st.caption("Convergence Check Δ: N/A")

def page_plots():
    """Main page for displaying interactive Plotly diagrams."""
    st.header("Structural Diagrams")

    if st.session_state.plot_data is None:
        st.warning("Please run the analysis first to generate plots.")
        return

    pdata = st.session_state.plot_data

    # SFD and BMD
    col_sfd, col_bmd = st.columns(2)
    with col_sfd:
        st.plotly_chart(create_diagram_plot(np.array(pdata['x_plot']), np.array(pdata['V_plot']), "Shear Force Diagram (ULS)", "Shear Force (V)", "kN", '#007bff', False), use_container_width=True)
    with col_bmd:
        st.plotly_chart(create_diagram_plot(np.array(pdata['x_plot']), np.array(pdata['M_plot']), "Bending Moment Diagram (ULS)", "Moment (M)", "kN-m", '#dc3545', True), use_container_width=True)

    st.divider()

    # Deflection and Curvature
    col_def, col_cur = st.columns(2)
    with col_def:
        # Deflection: negative downward values are negated for plotting as positive downward
        st.plotly_chart(create_diagram_plot(np.array(pdata['x_nodes']), -np.array(pdata['v_deflection']), "Deflection Curve (SLS)", "Deflection (v)", "mm", '#ffc107', False), use_container_width=True)
    with col_cur:
        # Convert curvature from 1/mm to 1/m by multiplying by 1000
        st.plotly_chart(create_diagram_plot(np.array(pdata['x_plot']), np.array(pdata['curvature']) * 1000.0, "Curvature Diagram (SLS)", "Curvature (1/m)", "1/m", '#6f42c1', False), use_container_width=True)

def page_report():
    """Main page for generating and downloading reports."""
    st.header("Design Reports & Downloads")

    if st.session_state.design_results is None:
        st.error("Run the analysis first to generate reports.")
        return

    inputs = st.session_state.inputs
    design_res = st.session_state.design_results
    fem_res = st.session_state.fem_results

    w_factored = inputs.get('gamma_f_dl', 1.5) * (inputs.get('w_dl_total', 0.0) + inputs.get('w_ll', 0.0))

    # --- Attempt to Generate PDF Data ---
    pdf_data = None
    plots_png = {}

    try:
        # 1. Generate PNG data for report (relying on backend Matplotlib calls)
        plots_png = plots.plot_results(inputs.get('L_eff', 3500.0), w_factored, fem_res.get('fem_results', {}), fem_res.get('I_e_data', {}))
    except Exception:
        st.warning("Could not generate backend Matplotlib plots for PDF report. PDFs will be text-only or limited.")

    try:
        # 2. Call the PDF report generator
        pdf_data = report.create_pdf_report(inputs, design_res, fem_res, plots_png)
    except Exception as e:
        st.error(f"""
        PDF Report Generation FAILED (Backend Error).
        Detail: {type(e).__name__}: {e}
        Action: The error originates in `report.py`. Please review and fix `report.py`'s style definitions or dependencies. CSV download still available.
        """)
        # Allow CSV generation even if PDF failed
        pdf_data = None

    # --- Output Downloads ---
    st.subheader("CSV Summary")
    try:
        csv_data = report.create_csv_summary(inputs, design_res, fem_res)
        st.download_button(label="⬇️ Download Design Summary (CSV)", data=csv_data, file_name="beam_design_summary.csv", mime="text/csv")
    except Exception:
        st.warning("CSV summary not available due to backend error.")

    st.subheader("PDF Detailed Report")
    if pdf_data:
        st.download_button(label="📄 Download Detailed Report (PDF)", data=pdf_data, file_name="beam_design_report.pdf", mime="application/pdf")
        st.markdown("---")
        st.subheader("Inline PDF Preview (First Page)")
        st.download_button(label="View PDF Preview", data=pdf_data, file_name="preview.pdf", mime="application/pdf")
    else:
        st.warning("PDF data could not be generated due to the backend error.")

def page_settings():
    """Sidebar page for general settings and disclaimers."""
    st.header("Settings & Disclaimer")

    st.session_state.analysis_mode = st.radio(
        "Analysis Mode",
        ['(B) Detailed Limit-State', '(A) Quick/Empirical'],
        index=0 if st.session_state.analysis_mode == '(B) Detailed Limit-State' else 1,
        key='mode_toggle',
        help="Detailed mode runs the full FEM analysis and convergence checks."
    )

    st.header("Assumptions & Code")
    st.info("""
    This application adheres to **IS 456:2000** for all ULS (Flexure/Shear) and SLS (Deflection) checks.

    **Key Assumptions:**
    - Loads are UDL only.
    - Beam is Simply Supported.
    - Deflection limit is conservatively set to $L/250$.
    - Effective Moment of Inertia ($I_e$) is calculated using the **Branson's Formula** interpretation.
    """)
    st.error("**Disclaimer**: This is a design aid. It is **not a substitute** for professional engineering judgment or licensed engineer sign-off.")

# --- Main Application Logic ---

def main():
    inject_css()

    # --- Header (Always Visible) ---
    st.markdown(f'<div style="color: {ACCENT_COLOR}; font-size: 24px; font-weight: bold;">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.caption(f'{APP_SUBTITLE} | Version {APP_VERSION} | {APP_AUTHOR}')
    st.divider()

    # --- Sidebar Navigation & Controls ---
    with st.sidebar:
        st.markdown(f'<h2 style="color: {ACCENT_COLOR};">Design Controls</h2>', unsafe_allow_html=True)

        # Navigation radio: ensure the current page value maps to a valid index
        pages = ["1. Inputs", "2. Design Results", "3. Plots & Diagrams", "4. Reports & Downloads", "5. Settings"]
        try:
            default_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
        except Exception:
            default_index = 0

        st.session_state.page = st.radio("Navigation", pages, index=default_index)

        st.divider()

        # Run Button (simple)
        if st.button("🚀 Run Analysis"):
            # Basic validation
            if st.session_state.inputs.get('L_eff', 0.0) < 1000.0 or st.session_state.inputs.get('b', 0.0) < 10.0 or st.session_state.inputs.get('fck', 0.0) < 10.0:
                st.error("Invalid inputs. Check dimensions/materials.")
            else:
                run_full_analysis()
                # Navigate to results after successful analysis
                st.session_state.page = "2. Design Results"
                st.experimental_rerun()

        st.divider()
        st.write("Current Mode: " + st.session_state.analysis_mode)

    # --- Main Content Renderer ---
    if st.session_state.page == "1. Inputs":
        page_inputs()
    elif st.session_state.page == "2. Design Results":
        page_results()
    elif st.session_state.page == "3. Plots & Diagrams":
        page_plots()
    elif st.session_state.page == "4. Reports & Downloads":
        page_report()
    elif st.session_state.page == "5. Settings":
        page_settings()
    else:
        st.info("Select a page from the sidebar.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: gray; font-size: 10px;'>Powered by Python, Streamlit, and Structural Engineering Logic. Code: saatwik.structures</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
  
