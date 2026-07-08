import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Set page configuration to wide layout for side-by-side plots
st.set_page_config(layout="wide", page_title="Metasurface Dashboard")

# Custom CSS styling to mirror your clean, white-and-slate aesthetic
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        div[data-testid="stForm"] { background-color: #f9f9fb; border-radius: 8px; border: 1px solid #e6e6f2; }
        h1, h2, h3 { color: #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- Core Physics Engine Functions ---
def acoustic_impedance_mf(sigma, d0, t, f, rho, zf, v_a, m, s, eta):
    omega = 2 * np.pi * f
    TL = np.zeros(len(f))
    
    for i in range(len(f)):
        ZH = []
        for j in range(len(d0)):
            # Maa Impedance Model
            X0 = (d0[j] / 2.0) * np.sqrt(omega[i] * rho / v_a)
            Zh_R = (32 * v_a * t) / (d0[j]**2) * (np.sqrt(1 + (X0**2 / 32.0)) + (np.sqrt(2.0) * X0 / 8.0) * (d0[j] / t))
            Zh_I = 1j * rho * omega[i] * t * (1 + (9 + (X0**2) / 2.0)**(-0.5) + ((8.0 / (3.0 * np.pi)) * (d0[j] / t)))
            ZH.append(Zh_R + Zh_I)
            
        ZH = np.array(ZH)
        ZH_eff_1 = np.sum(sigma[-1] / ZH[-1])
        zeff = 1.0 / ZH_eff_1
        
        # Panel Mechanical Impedance
        zp = eta * np.sqrt(s * m) + 1j * ((omega[i] * m) - (s / omega[i]))
        zq = zp + ((ZH[0] * ZH[1]) / zeff)
        
        # Radiation Impedance
        gamma = 1.0 - (ZH[0] / zeff)
        delta = 1.0 - np.sum(sigma) + (ZH[0] / zeff)
        zr = zeff / (1.0 + ((gamma * delta) * (zeff / zq)))
        
        # Transmission Loss Calculation
        tau = np.abs(2.0 * zf / (zr + 2.0 * zf))**2
        TL[i] = 10.0 * np.log10(1.0 / tau)
        
    return TL

# --- Application Layout ---
st.title("Acoustic & Aerodynamic Geometric Dashboard")

# --- Dropdown Menu (Geometry Selector) ---
geom_choice = st.selectbox(
    "Select Geometry:",
    options=["Geometry 1 (2 holes)", "Geometry 2 (3 holes)", "Geometry 3 (4 holes)"],
    index=1  # Default to Geometry 2 like your MATLAB app
)

# Parse selection index for default conditional values
if "2 holes" in geom_choice:
    default_t, default_ns, default_a, default_b,default_dc,default_airV,default_holden,default_ds  = 24.0, 2, 0.23, 0.22, 13.37, 2.8, 9, 13.37/np.sqrt(9)
elif "3 holes" in geom_choice:
    default_t, default_ns, default_a, default_b,default_dc,default_airV,default_holden,default_ds  = 33.0, 3, 0.38, 0.39,17.14, 2.8, 10, 17.14/np.sqrt(10)
else:
    default_t, default_ns, default_a, default_b,default_dc,default_airV,default_holden,default_ds  = 42.0, 4, 0.19, 0.33, 10.42, 2.8, 10, 10.42/np.sqrt(10)

# --- Control Parameters Container (Using Sidebar for clean space) ---
st.sidebar.header("Control Parameters")

# Sidebar Parameter Inputs (Combining input fields and sliders natively)
a_space = st.sidebar.slider("a_space (mm):", 0.1, 5.0, default_a, step=0.01)
b_space = st.sidebar.slider("b_space (mm):", 0.1, 4.0, default_b, step=0.01)
d_central_mm = st.sidebar.slider("Center Hole Dia (mm):", 3.0, 25.0, default_dc, step=0.1)
hole_den_v = st.sidebar.slider("side hole dia factor:", 2, 15, default_holden, step=1)

if hole_den_v != default_holden:
    st.sidebar.warning("Hole dia factor(metacond) changed from default; side hole diameter will be recalculated.")
    d_sidehole_mm = st.sidebar.slider("Side Hole(s) Dia (mm):", 3.0, 25.0, (1.0 / np.sqrt(hole_den_v)) * d_central_mm, step=0.1)
else:
    d_sidehole_mm = st.sidebar.slider("Side Hole(s) Dia (mm):", 3.0, 25.0, default_ds, step=0.1)

thickness_mm = st.sidebar.slider("Thickness (mm):", 20.0, 55.0, default_t, step=1.0)
n_holes_s = st.sidebar.slider("Number of Side Holes:", 2, 4, default_ns, step=1)
air_velocity = st.sidebar.slider("Air Velocity (m/s):", 1.0, 20.0, default_airV, step=0.1)



# Frequency Array Setup (Simulating your external .mat vector out-of-the-box)
freq = np.array([
    0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0,
    12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
    125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
])

# Material Constants
rho = 1.225
c = 343.0
zf = rho * c
v_a = 1.8e-5
s = 3e9
eta = 0.005
rho_panel = 1180.0

# --- Math Calculations & Parameter Setup ---
d_central = d_central_mm * 1e-3
t = thickness_mm * 1e-3
vel_p = air_velocity
a_space_m = a_space * 1e-3
b_space_m = b_space * 1e-3
d_side = d_sidehole_mm * 1e-3

if hole_den_v != default_holden:
    d_side = (1.0 / np.sqrt(hole_den_v)) * d_central

d_list = [d_central, d_side]
N_holes = [1, n_holes_s]
apothem_panel = (d_central / 2.0) + 2.0 * (d_side / 2.0) + a_space_m + b_space_m

# Geometric Area Logic
if n_holes_s == 2:
    A_p = 2.0 * np.sqrt(3.0) * (apothem_panel**2)
elif n_holes_s == 3:
    A_p = 9.0 * np.tan(np.radians(20.0)) * (apothem_panel**2)
elif n_holes_s == 4:
    A_p = 12.0 * np.tan(np.radians(15.0)) * (apothem_panel**2)

sigma_central = (N_holes[0] * (np.pi * (d_central**2) / 4.0)) / A_p
sigma_side = (N_holes[1] * (np.pi * (d_side**2) / 4.0)) / A_p
sigma = np.array([sigma_central, sigma_side])

# --- Run Calculators ---
TL_perf = acoustic_impedance_mf(sigma, d_list, t, freq, rho, zf, v_a, rho_panel * t, s, eta)

k_factor = (0.707 * (1.0 - np.sum(sigma))**0.375 + 1.0 - np.sum(sigma))**2 * (1.0 / np.sum(sigma)**2)
delta_P_1 = k_factor * (rho * vel_p**2 / 2.0)
delta_P = 0.0177 * (delta_P_1**2) + 0.7359 * delta_P_1

# --- Render Plots Side-by-Side ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Insertion Loss")
    fig_tl, ax_tl = plt.subplots(figsize=(6, 4.5))
    ax_tl.plot(freq, TL_perf, 'b-', linewidth=2)
    ax_tl.set_xscale('log')
    ax_tl.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax_tl.set_ylabel('IL (dB)', fontweight='bold')
    ax_tl.set_ylim([0, 50])
    ax_tl.grid(True, which="both", alpha=0.4)
    
    # Safe index annotation bounding check
    ann_idx = min(31, len(TL_perf) - 1)
    ax_tl.text(0.05, 0.9, f"Insertion Loss: {TL_perf[ann_idx]:.1f} dB @ {freq[ann_idx]:.0f} Hz",
               transform=ax_tl.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(facecolor='#f5f5f5', edgecolor='black', boxstyle='round,pad=0.5'))
    st.pyplot(fig_tl)

with col2:
    st.subheader("Aerodynamic Pressure Drop")
    fig_pd, ax_pd = plt.subplots(figsize=(6, 4.5))
    ax_pd.bar(['Delta P'], [delta_P], color ='#d35400', width=0.4)
    ax_pd.set_ylabel('Delta P (Pa)', fontweight='bold')
    ax_pd.set_ylim([0, max(100.0, delta_P * 1.2)])
    ax_pd.grid(axis='y', alpha=0.4)
    
    ax_pd.text(0, delta_P + max(2.0, delta_P * 0.05), f"{delta_P:.2f} Pa",
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    st.pyplot(fig_pd)