import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Task 5.2: DAB Power Flow vs Phase Shift
print("=== TASK 5.2: DAB Power Flow Analysis ===\n")

# Given parameters from Table II
V_in = 700         # Input voltage [V]
V_out = 700        # Output voltage [V]
f_sw = 1000        # Switching frequency [Hz]
L = 1e-3           # Inductance [H]

# Phase shift range from 0 to 1
D = np.linspace(0, 1, 1000)  # Phase shift ratio

# Power flow equation for DAB
P_dab = (V_in * V_out * D * (1 - np.abs(D))) / (2 * f_sw * L)

# Convert to kW for better readability
P_dab_kW = P_dab / 1000

# Find maximum power
P_max = np.max(P_dab_kW)
D_max = D[np.argmax(P_dab_kW)]

print(f"DAB Parameters:")
print(f"V_in = {V_in} V, V_out = {V_out} V")
print(f"f_sw = {f_sw} Hz, L = {L} H")
print(f"Maximum power: {P_max:.2f} kW at D = {D_max:.3f}\n")

def power_equation(D, P_target_kW):
    """Equation to solve for D given target power"""
    P_target_W = P_target_kW * 1000
    return (V_in * V_out * D * (1 - D)) / (2 * f_sw * L) - P_target_W

# Find phase shift values for 50 kW power flow
P_target = 50  # Target power in kW

# Method 1: Using scipy.optimize.fsolve
D_initial_guesses = [0.2, 0.8]  # Initial guesses for the two solutions
D_solutions = []

for guess in D_initial_guesses:
    solution = fsolve(power_equation, guess, args=(P_target,))
    if 0 <= solution[0] <= 1:
        D_solutions.append(solution[0])

# Method 2: Quadratic equation solution
constant = (P_target * 1000 * 2 * f_sw * L) / (V_in * V_out)
# Equation: D^2 - D + constant = 0
a, b, c = 1, -1, constant
discriminant = b**2 - 4*a*c
D_quadratic = [(-b + np.sqrt(discriminant))/(2*a), (-b - np.sqrt(discriminant))/(2*a)]
D_quadratic = [d for d in D_quadratic if 0 <= d <= 1]

# Use the solutions from quadratic method (more accurate)
D_valid = sorted(D_quadratic)

print(f"Target power: {P_target} kW")
print(f"\nPhase shift values for {P_target} kW power flow:")
for i, d_val in enumerate(D_valid, 1):
    phase_angle_deg = d_val * 180
    print(f"D{i} = {d_val:.3f} ({phase_angle_deg:.1f}° phase shift)")

# Calculate actual power at these points for verification
P_actual = (V_in * V_out * np.array(D_valid) * (1 - np.array(D_valid))) / (2 * f_sw * L)
print(f"\nVerification - Actual power at these points:")
for i, d_val in enumerate(D_valid):
    print(f"At D={d_val:.3f}: P = {P_actual[i]/1000:.2f} kW")

# Create the power flow plot
plt.figure(figsize=(12, 8))

# Main power flow curve
plt.plot(D, P_dab_kW, 'b-', linewidth=2, label='Power Flow')

# Mark the 50 kW points
plt.plot(D_valid, [P_target] * len(D_valid), 'ro', markersize=8, 
         markerfacecolor='red', label=f'{P_target} kW Operating Points')

# Mark maximum power point
plt.plot(D_max, P_max, 'gs', markersize=10, markerfacecolor='green', 
         label=f'Maximum Power Point (D={D_max:.2f})')

# Add labels and grid
plt.xlabel('Phase Shift Ratio D', fontsize=12)
plt.ylabel('Power Flow [kW]', fontsize=12)
plt.title('DAB Power Flow vs Phase Shift Ratio', fontsize=14)
plt.grid(True, alpha=0.3)

# Add reference lines
plt.axvline(x=D_valid[0], color='red', linestyle='--', alpha=0.7)
plt.text(D_valid[0], P_target*0.8, f'D={D_valid[0]:.3f}', 
         ha='center', va='bottom', color='red')

if len(D_valid) > 1:
    plt.axvline(x=D_valid[1], color='red', linestyle='--', alpha=0.7)
    plt.text(D_valid[1], P_target*0.8, f'D={D_valid[1]:.3f}', 
             ha='center', va='bottom', color='red')

plt.axvline(x=D_max, color='green', linestyle='--', alpha=0.7)
plt.text(D_max, P_max*0.9, f'D={D_max:.2f}\n(P_max)', 
         ha='center', va='top', color='green')

plt.axhline(y=P_target, color='black', linestyle='--', alpha=0.5)
plt.text(0.05, P_target, f'{P_target} kW', 
         ha='left', va='bottom', color='black')

plt.legend(loc='upper right')
plt.tight_layout()

# Additional Analysis
print("\n" + "="*50)
print("ADDITIONAL ANALYSIS")
print("="*50)

# Calculate power at specific points
test_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\nPower at specific phase shift values:")
print("D Value | Power [kW] | Phase Angle")
print("-" * 35)
for d in test_points:
    power = (V_in * V_out * d * (1 - d)) / (2 * f_sw * L * 1000)
    angle = d * 180
    print(f"{d:7.2f} | {power:10.2f} | {angle:11.1f}°")

# Efficiency considerations
print(f"\n--- Efficiency Considerations ---")
if len(D_valid) >= 2:
    print(f"Operating at D={D_valid[0]:.3f} is preferred over D={D_valid[1]:.3f} because:")
    print(f"- Lower phase shift reduces circulating currents")
    print(f"- Smaller reactive power component")
    print(f"- Higher efficiency and lower losses")
    print(f"- Reduced stress on semiconductor devices")

# Create a detailed analysis plot
plt.figure(figsize=(14, 10))

# Subplot 1: Main power curve
plt.subplot(2, 2, 1)
plt.plot(D, P_dab_kW, 'b-', linewidth=3)
plt.plot(D_valid, [P_target] * len(D_valid), 'ro', markersize=10, markerfacecolor='red')
plt.plot(D_max, P_max, 'gs', markersize=12, markerfacecolor='green')
plt.xlabel('Phase Shift Ratio D')
plt.ylabel('Power [kW]')
plt.title('DAB Power Transfer Characteristics')
plt.grid(True, alpha=0.3)
plt.legend(['Power Flow', f'{P_target} kW Points', f'P_max = {P_max:.1f} kW'])

# Subplot 2: Zoom around 50 kW points
plt.subplot(2, 2, 2)
plt.plot(D, P_dab_kW, 'b-', linewidth=2)
plt.plot(D_valid, [P_target] * len(D_valid), 'ro', markersize=8, markerfacecolor='red')
plt.xlim(0, 1)
plt.ylim(0, P_max * 1.1)
plt.xlabel('Phase Shift Ratio D')
plt.ylabel('Power [kW]')
plt.title('Full Range View')
plt.grid(True, alpha=0.3)

# Subplot 3: Phase shift in degrees
plt.subplot(2, 2, 3)
phase_angle_deg = D * 180
plt.plot(phase_angle_deg, P_dab_kW, 'purple', linewidth=2)
plt.plot([d*180 for d in D_valid], [P_target] * len(D_valid), 'ro', 
         markersize=8, markerfacecolor='red')
plt.xlabel('Phase Shift Angle [degrees]')
plt.ylabel('Power [kW]')
plt.title('Power vs Phase Angle (Degrees)')
plt.grid(True, alpha=0.3)

# Subplot 4: Normalized power
plt.subplot(2, 2, 4)
P_normalized = P_dab_kW / P_max
plt.plot(D, P_normalized, 'orange', linewidth=2)
plt.plot(D_valid, [P_target/P_max] * len(D_valid), 'ro', 
         markersize=8, markerfacecolor='red')
plt.xlabel('Phase Shift Ratio D')
plt.ylabel('Normalized Power (P/P_max)')
plt.title('Normalized Power Transfer')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Print comprehensive results summary
print("\n" + "="*60)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*60)
print(f"{'Parameter':<20} {'Value':<15} {'Units':<10}")
print("-" * 50)
print(f"{'Input Voltage':<20} {V_in:<15.0f} {'V':<10}")
print(f"{'Output Voltage':<20} {V_out:<15.0f} {'V':<10}")
print(f"{'Switching Freq':<20} {f_sw:<15.0f} {'Hz':<10}")
print(f"{'Inductance':<20} {L*1000:<15.3f} {'mH':<10}")
print(f"{'Maximum Power':<20} {P_max:<15.2f} {'kW':<10}")
print(f"{'Optimal D for P_max':<20} {D_max:<15.3f} {'':<10}")
print(f"{'Target Power':<20} {P_target:<15.0f} {'kW':<10}")
for i, d_val in enumerate(D_valid, 1):
    print(f"{f'Solution D{i}':<20} {d_val:<15.3f} {'':<10}")
    print(f"{f'Phase Angle D{i}':<20} {d_val*180:<15.1f} {'degrees':<10}")

# Show all plots
plt.show()

# Export data for further analysis (optional)
print(f"\nData exported for {len(D)} data points from D=0 to D=1")
print("Use D and P_dab_kW arrays for further analysis")

# Final recommendation
print("\n" + "="*50)
print("DESIGN RECOMMENDATION")
print("="*50)
if len(D_valid) >= 2:
    recommended_D = min(D_valid, key=lambda x: abs(x - 0.5))
    print(f"Recommended operating point: D = {D_valid[0]:.3f}")
    print("Reasons:")
    print("1. Lower phase shift reduces circulating currents")
    print("2. Higher efficiency operation")
    print("3. Reduced semiconductor stress")
    print("4. Better dynamic response")
    print(f"5. Phase angle of {D_valid[0]*180:.1f}° is more practical")