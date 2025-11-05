import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------ load data ------------------
csv_path = "/Users/42d/Downloads/All_Data.csv"   # columns: alpha_deg  s=x/λ_D  phi_over_Te
alpha, s, phi = np.loadtxt(csv_path, unpack=True)
m = np.isfinite(alpha) & np.isfinite(s) & np.isfinite(phi)
alpha, s, phi = alpha[m], np.clip(s[m], 0.0, None), phi[m]

# ------------------ sign-safe two-exp model ------------------
# φ = -[ B1(α) e^{-k1(α) s} + B2(α) e^{-k2(α) s} ],   B1,B2,k1,k2 > 0
def unpack(p):
    b10, b11, b20, b21, p10, p11, p20, p21 = p
    return b10, b11, b20, b21, p10, p11, p20, p21

def B1(a, b10, b11): return np.exp(b10 + b11*a)
def B2(a, b20, b21): return np.exp(b20 + b21*a)
def k1(a, p10, p11): return np.exp(p10 + p11*a)
def k2(a, p20, p21): return np.exp(p20 + p21*a)

def phi_model(p, a, s):
    b10, b11, b20, b21, p10, p11, p20, p21 = unpack(p)
    return -(B1(a,b10,b11)*np.exp(-k1(a,p10,p11)*s) +
             B2(a,b20,b21)*np.exp(-k2(a,p20,p21)*s))

def residuals(p, a, s, y):
    return phi_model(p, a, s) - y

# ------------------ initial guesses ------------------
b10_0, b11_0 = np.log(1.3),  0.0   # amplitudes ~ O(1)
b20_0, b21_0 = np.log(1.3),  0.0
p10_0, p11_0 = np.log(0.02), 0.0   # slow tail
p20_0, p21_0 = np.log(0.3),  0.0   # near-wall
p0 = np.array([b10_0,b11_0,b20_0,b21_0,p10_0,p11_0,p20_0,p21_0])

# ------------------ fit ------------------
res = least_squares(residuals, p0, args=(alpha, s, phi),
                    loss="soft_l1", f_scale=0.1, max_nfev=40000)
b10,b11,b20,b21,p10,p11,p20,p21 = res.x
rms = np.sqrt(np.mean(res.fun**2))
print(f"RMS residual: {rms:.4e}")

# ------------------ plot: data vs fit ------------------
unique_a = np.unique(alpha)
take = np.unique(np.clip((np.array([0,0.25,0.5,0.75,1.0])*(len(unique_a)-1)).round().astype(int),
                         0, len(unique_a)-1))
sample_alphas = unique_a[take]

s_plot = np.linspace(0, 80, 600)
plt.figure(figsize=(8,6))
for a in sample_alphas:
    msk = np.isclose(alpha, a)
    plt.scatter(s[msk], phi[msk], s=14, alpha=0.6, label=f"data α={a:g}°")
    yfit = phi_model(res.x, a*np.ones_like(s_plot), s_plot)
    plt.plot(s_plot, yfit, linewidth=2)
plt.xlabel("Distance from wall / λ_D")
plt.ylabel("Sheath potential φ / T_e")
plt.title("Sign-safe two-exponential fit: φ = -[B1 e^{-k1 s} + B2 e^{-k2 s}]")
plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

def E_model_norm(p, a, s):
    # E normalized by Te/λ_D:  E_norm = B1*k1*exp(-k1 s) + B2*k2*exp(-k2 s)
    b10, b11, b20, b21, p10, p11, p20, p21 = unpack(p)
    B1v = B1(a, b10, b11)
    B2v = B2(a, b20, b21)
    K1v = k1(a, p10, p11)
    K2v = k2(a, p20, p21)
    return B1v*K1v*np.exp(-K1v*s) + B2v*K2v*np.exp(-K2v*s)

# ---- separate figure for E (normalized by Te/λ_D) ----
plt.figure(figsize=(8,6))
for a in sample_alphas:
    Enorm = E_model_norm(res.x, a*np.ones_like(s_plot), s_plot)
    plt.plot(s_plot, Enorm, linewidth=2, label=f"α={a:g}°")
plt.ylim(0,0.25)
plt.xlabel("Distance from wall / λ_D")
plt.ylabel("E normalized (units of 1/λ_D)")
plt.title("Sheath electric field (normalized):  E / (Te/λ_D)")
plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

#
## ------------------ emit valid C++ (braces escaped) ------------------
#cpp = (
#    "#include <cmath>\n"
#    "\n"
#    "// φ/Te = -[ B1(α) e^(-k1(α) s) + B2(α) e^(-k2(α) s) ],  s = x/λ_D\n"
#    "// Parameterization (α in DEGREES):\n"
#    "//   B1(α) = exp(b10 + b11*α),  B2(α) = exp(b20 + b21*α)\n"
#    "//   k1(α) = exp(p10 + p11*α),  k2(α) = exp(p20 + p21*α)\n"
#    "inline double B1_of_alpha(double alpha_deg) {{\n"
#    "    return std::exp({b10:.12g} + {b11:.12g}*alpha_deg);\n"
#    "}}\n"
#    "inline double B2_of_alpha(double alpha_deg) {{\n"
#    "    return std::exp({b20:.12g} + {b21:.12g}*alpha_deg);\n"
#    "}}\n"
#    "inline double k1_of_alpha(double alpha_deg) {{\n"
#    "    return std::exp({p10:.12g} + {p11:.12g}*alpha_deg);\n"
#    "}}\n"
#    "inline double k2_of_alpha(double alpha_deg) {{\n"
#    "    return std::exp({p20:.12g} + {p21:.12g}*alpha_deg);\n"
#    "}}\n"
#    "\n"
#    "inline double sheath_phi_over_Te(double alpha_deg, double s_norm) {{\n"
#    "    const double s = std::fabs(s_norm);\n"
#    "    const double B1 = B1_of_alpha(alpha_deg);\n"
#    "    const double B2 = B2_of_alpha(alpha_deg);\n"
#    "    const double K1 = k1_of_alpha(alpha_deg);\n"
#    "    const double K2 = k2_of_alpha(alpha_deg);\n"
#    "    // if (s > 200.0) return 0.0; // optional cutoff\n"
#    "    return -(B1*std::exp(-K1*s) + B2*std::exp(-K2*s));\n"
#    "}}\n"
#    "\n"
#    "// E normalized by Te/λ_D (sign per your convention)\n"
#    "inline double sheath_E_over_Te_per_lambdaD(double alpha_deg, double s_norm) {{\n"
#    "    const double s = std::fabs(s_norm);\n"
#    "    const double B1 = B1_of_alpha(alpha_deg);\n"
#    "    const double B2 = B2_of_alpha(alpha_deg);\n"
#    "    const double K1 = k1_of_alpha(alpha_deg);\n"
#    "    const double K2 = k2_of_alpha(alpha_deg);\n"
#    "    return (B1*K1)*std::exp(-K1*s) + (B2*K2)*std::exp(-K2*s);\n"
#    "}}\n"
#).format(b10=b10, b11=b11, b20=b20, b21=b21, p10=p10, p11=p11, p20=p20, p21=p21)
#
#print("\n/* ---- C++ you can paste ---- */")
#print(cpp)
#
