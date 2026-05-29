from typing import List

from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp


#===Constants===#
F = 0.0073 #[m3/s]
L_PIPE = 66 #[m]
D_PIPE = 0.072 #[m]
EPSILON = 1 * 10 ** -5 #[m]
NU = 5 * 10 ** -7 #[m2/s]
Z_PUMP= -0.5 #[m]
K = 0.75
g = 9.81 #[m/s2]
NUMBER_OF_TURNS = 5
RES_TURB = np.geomspace(4000, 10**8, num=50)
RES_LAM = np.geomspace(0.1, 4000, num=50)
eDs = [0.00001, 0.0001, 0.001]
CSV_FILE_IMPELLERS = 'lab1_impellers.csv'
CSV_FILE_EFF = 'lab1_eff1.csv'
P_OUT = 182385 #[Pa]
P_IN = 222915 #[Pa]
RHO = 777 #[kg/m3]
DELTA_HEIGHT = 32 #[m]
KINETIC_ALPHA = 1
NPSH_REQUIRED_BASE = 0.5761 #[m]
NPSH_REQUIRED_EXPONENT = 0.0511


#===================#
# Part A
#===================#


def Fanning(Re, e_D):
    Validation_Fanning_Range(Re, e_D)
    if Re < 4000:
        return 16 / Re
    else:
        return 1.375 * 10**(-3) * (1 + (2000 * e_D + 10**6 / Re)**(1/3))
 
def Print_Fanning():
    for e_D in eDs:
        plt.plot(RES_TURB, [Fanning(r, e_D) for r in RES_TURB], label=f'e/D={e_D}')
    plt.plot(RES_LAM, [Fanning(r, 0) for r in RES_LAM], label='e/D=0', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Fanning Friction Factor (f)')
    plt.title('Fanning Friction Factor vs Reynolds Number')
    plt.legend()
    plt.grid()
    plt.show()

def hLT(): # Calculate the head loss using the Fanning friction factor [m^2/s2]
    Re = F * D_PIPE / (math.pi * (D_PIPE/2)**2 * NU)
    e_D = EPSILON / D_PIPE
    f = Fanning(Re, e_D)
    v = F / (math.pi * (D_PIPE/2)**2)
    hLT = (f * (L_PIPE / D_PIPE) + K*NUMBER_OF_TURNS) * 2 * v**2
    return hLT

def Validation_Fanning_Range(Re, e_D):
    if 0 > Re or Re > 100000000:
        raise ValueError("Re should be between 0 and 10^8")
    if 0 > e_D or e_D > 0.01:
        raise ValueError("e/D should be between 0 and 0.01")
    


#===================#
# Part B
#===================#

def Extract_Parameters_From_CSV_Impellers(impeller_index=0):
    """
    Load CSV file, extract first 2 columns.
    """
    data = np.genfromtxt(CSV_FILE_IMPELLERS, delimiter=',', skip_header=0)
    
    name = data[0, 1+2*impeller_index]  # Name of the impeller (not used in calculations)
    X = data[:, 0+2*impeller_index]  # Flow rate [m3/hr]
    Y = data[:, 1+2*impeller_index]  # Head [m]

    return name, X, Y

def System_Head(Q: List[float]):
    h_LT = hLT()
    v_ave_in = 0
    H_system = []
    for q in Q:
        v_ave_out = q / (math.pi * (D_PIPE/2)**2)
        H_system.append((P_OUT - P_IN) / (RHO * g) + DELTA_HEIGHT + (KINETIC_ALPHA * (v_ave_out*2 - v_ave_in*2) / (2 * g) + (h_LT / g)))

    func = np.poly1d(np.polyfit(Q, H_system, 2))

    return func

def Parabolic_Approximation(X, Y):
    """
    Fit a parabolic approximation to the given data.
    """
    # Fit a parabolic (2nd degree polynomial) approximation: Y = a*X^2 + b*X + c
    coefficients = np.polyfit(X, Y, 2)    
    poly_func = np.poly1d(coefficients)
    
    return poly_func

def Find_Intersection(func1, func2, x0=0):
    """
    Find the intersection point of two functions using fsolve.
    """
    def difference(x):
        return func1(x) - func2(x)
    
    intersection_x = fsolve(difference, x0)[0]
    intersection_y = func1(intersection_x)
    
    return intersection_x, intersection_y

def Plot_All(funcs: List[np.poly1d], Dots: List[tuple]):
    """
    Plot the original data points and the fitted functions.
    """
    fig, ax = plt.subplots()

    for i in range(4):
        name, X, Y = Extract_Parameters_From_CSV_Impellers(impeller_index=i)
        ax.scatter(X, Y, color='red', label=name)

    x_fit = np.linspace(min(X), max(X), 100)
    for i, func in enumerate(funcs):
        y_fit = func(x_fit)
        ax.plot(x_fit, y_fit, color='green', label=f'Fitted Function {i+1}')
    for i, dot in enumerate(Dots):
        ax.scatter(dot[0], dot[1], color='black', label=f'Intersection Point {i+1}', zorder=5)

    ax.set_xlabel('Flow Rate (m3/hr)')
    ax.set_ylabel('Head (m)')
    ax.set_title('Pump Performance Curve')

    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x / 3600, lambda x: x * 3600))
    ax_top.set_xlabel('Flow Rate (m3/s)')

    ax.legend()
    ax.grid()
    plt.show()





#===================#
# Part C
#===================#


def NPSH_Required(Q):
    NPSH_required = NPSH_REQUIRED_BASE * math.exp(NPSH_REQUIRED_EXPONENT * Q)
    return NPSH_required

def NPSH_Available():
    NPSH_available = (P_IN - P_OUT) / (RHO * g) + DELTA_HEIGHT + (0.25 * (hLT() / g))
    return NPSH_available

def Extract_Parameters_From_CSV_Eff(eff_index=0):
    """
    Load CSV file, extract first 2 columns.
    """
    data = np.genfromtxt(CSV_FILE_EFF, delimiter=',', skip_header=0)
    
    name = data[0, 1+2*eff_index]  # Name of the efficiency curve (not used in calculations)
    X = data[:, 0+2*eff_index]  # Flow rate [m3/hr]
    Y = data[:, 1+2*eff_index]  # Head [m]

    return name, X, Y