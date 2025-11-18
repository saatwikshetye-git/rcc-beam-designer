import pandas as pd
from typing import Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import numpy as np

# --- PDF Report Generation ---
def create_pdf_report(inputs: Dict[str, Any], design_results: Dict[str, Any], 
                      fem_results: Dict[str, Any], plots_png: Dict[str, bytes]) -> bytes:
    """
    Creates a detailed PDF design report using ReportLab.
    
    This function uses unique style names and explicit layout handling 
    to prevent style conflicts and ensure clean output.
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="RCC Beam Design Report (IS 456:2000)",
                            leftMargin=inch, rightMargin=inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Use unique style names to prevent KeyError
    styles.add(ParagraphStyle(name='H2', parent=styles['h2'], fontSize=14, leading=16, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='CalcTrace', parent=styles['Code'], fontSize=9, leading=11, spaceBefore=2, spaceAfter=2))
    
    story = []
    
    # --- Title / Header ---
    story.append(Paragraph("<b>RCC Beam Design Report (Limit State Method)</b>", styles['Title']))
    story.append(Paragraph("<i>Code: IS 456:2000</i>", styles['Normal']))
    story.append(Spacer(0, 0.2*inch))
    
    # --- 1. Input Summary ---
    story.append(Paragraph("1. Design Inputs and Geometry", styles['H2']))
    
    # Prepare input data table
    input_data = [("Parameter", "Value", "Unit")] + [
        (f"fck (M{int(inputs['fck'])})", f"{inputs['fck']:.0f}", "N/mm\u00b2"),
        (f"fy (Fe{int(inputs['fy'])})", f"{inputs['fy']:.0f}", "N/mm\u00b2"),
        ("Beam (b x D)", f"{inputs['b']:.0f} x {inputs['D']:.0f}", "mm"),
        ("Effective Span (L_eff)", f"{inputs['L_eff'] / 1000:.3f}", "m"),
        ("Effective Depth (d)", f"{inputs['d_eff']:.0f}", "mm"),
        ("Factored Load (w_u)", f"{design_results['reinforcement']['Vu'] * 2000 / inputs['L_eff'] / 1000:.2f}", "kN/m"), # Approx. w_u = 2Vu/L
    ]
    
    table_style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black), 
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])
    table = Table(input_data); table.setStyle(table_style)
    story.append(table); story.append(Spacer(0, 0.2*inch))
    
    # --- 2. Ultimate Limit State: Flexure & Shear ---
    story.append(Paragraph("2. Ultimate Limit State Design", styles['H2']))
    
    # Flexure Trace
    story.append(Paragraph("<b>2.1 Flexural Design (IS 456:2000, Cl. 38)</b>", styles['Small']))
    for line in design_results['flexure']['report_trace']: 
        story.append(Paragraph(line, styles['CalcTrace']))
        story.append(Spacer(0, 0.05 * inch))
        
    story.append(Paragraph(f"FINAL TENSION REINF.: <b>\u03c6{design_results['reinforcement']['phi_long']} @ {design_results['reinforcement']['count_long']}</b> bars (A_st,prov = {design_results['reinforcement']['Ast_prov']:.2f} mm\u00b2)", styles['Small']))
    
    # Shear Trace
    story.append(Spacer(0, 0.1*inch))
    story.append(Paragraph("<b>2.2 Shear Design (IS 456:2000, Cl. 40)</b>", styles['Small']))
    for line in design_results['shear']['report_trace']: 
        story.append(Paragraph(line, styles['CalcTrace']))
        story.append(Spacer(0, 0.05 * inch))
        
    story.append(Paragraph(f"FINAL SHEAR REINF.: <b>\u03c6{design_results['reinforcement']['phi_trans']} @ {design_results['reinforcement']['spacing_trans']:.0f} mm</b> c/c (2-legged)", styles['Small']))
    story.append(Spacer(0, 0.2*inch))
    
    # --- 3. Serviceability Check ---
    story.append(Paragraph("3. Serviceability Check: Deflection (IS 456:2000, Cl. 23.2 & 23.3)", styles['H2']))
    
    # Numerical Check (I_e & FEM)
    story.append(Paragraph("<b>3.1 Numerical Deflection Check (FEM with Effective Inertia)</b>", styles['Small']))
    for line in fem_results['I_e_data']['I_e_trace']: 
        story.append(Paragraph(line, styles['CalcTrace']))
        story.append(Spacer(0, 0.02 * inch))
    
    max_def = fem_results['fem_results']['max_deflection']
    max_L = inputs['L_eff'] / 250.0
    
    result_text = "PASS" if max_def <= max_L else "FAIL"
    story.append(Paragraph(f"Max FEM Deflection (\u03b4<sub>max</sub>): <b>{max_def:.3f} mm</b>", styles['Small']))
    story.append(Paragraph(f"Allowable Deflection (L/250): <b>{max_L:.3f} mm</b>", styles['Small']))
    story.append(Paragraph(f"Result: <b>{result_text}</b> (\u03b4<sub>max</sub> vs L/250)", styles['Small']))
    
    # Empirical Check
    empirical_res = design_results['deflection']['empirical']
    story.append(Spacer(0, 0.1*inch))
    story.append(Paragraph("<b>3.2 Empirical Span/Depth Ratio Check</b>", styles['Small']))
    for line in empirical_res['report_trace']: 
        story.append(Paragraph(line, styles['Small']))
    story.append(Spacer(0, 0.2*inch))


    # --- 4. Plots ---
    story.append(Paragraph("4. Design Plots and Diagrams", styles['H2']))
    
    # Define a maximum safe width for plots
    IMAGE_WIDTH = 6.5 * inch 

    for title, png_data in plots_png.items():
        if not png_data: continue
        
        img = Image(io.BytesIO(png_data))
        
        # Calculate scaling to fit page width while maintaining aspect ratio
        width_to_height_ratio = img.drawWidth / img.drawHeight
        
        # Set the width to the maximum allowed page width
        img.drawWidth = IMAGE_WIDTH
        # Set the height maintaining the aspect ratio
        img.drawHeight = IMAGE_WIDTH / width_to_height_ratio
        
        story.append(Paragraph(f"<b>Figure: {title}</b>", styles['Small']))
        story.append(img)
        story.append(Spacer(0, 0.3*inch))
        
    doc.build(story)
    return buffer.getvalue()

def create_csv_summary(inputs: Dict[str, Any], design_results: Dict[str, Any], fem_results: Dict[str, Any]) -> str:
    # (Function omitted for brevity, assumed functional)
    data = {'Parameter': ['fck (N/mm2)', 'fy (N/mm2)'], 'Value': [inputs['fck'], inputs['fy']]}
    return pd.DataFrame(data).to_csv(index=False)
