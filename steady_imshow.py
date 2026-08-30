from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import os
import ufl


# =============================================================================
# Parameter sweep for the STEADY LEAKY-WALL MODEL
#
# Solve
#
#     C_T + U_eff(X) C_X = d/dX [D_eff(X) C_X] - U_eff,X(X) C
#
#
# For every pair (beta, Pe), calculate
#
#     C_max(beta, Pe) = max_{0 <= T <= T_final} max_X C(X,T).
#
# In other words, we record the largest concentration reached at ANY
# time during the simulation.
# =============================================================================


# =============================================================================
# 0. Plotting parameters
# =============================================================================

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

plt.rcParams["axes.formatter.use_mathtext"] = True


# =============================================================================
# 1. Parameter ranges
# =============================================================================

beta_values = np.linspace(0.1, 5.0, 40)

Pe_values = np.linspace(0.1, 100.0, 50)

epsilon_value = 0.002


# =============================================================================
# 2. Time discretisation
# =============================================================================

T_final = 50.0

num_steps = 3000

dt_value = T_final / num_steps


# =============================================================================
# 3. Mesh and finite-element space
# =============================================================================

nx = 500

mesh = IntervalMesh(nx, 0.0, 1.0)

W = FunctionSpace(mesh, "P", 1)

x = SpatialCoordinate(mesh)

X = x[0]


# Trial and test functions

C = TrialFunction(W)

q = TestFunction(W)


# =============================================================================
# 4. Boundary conditions
# =============================================================================

def left_boundary(point, on_boundary):
    return (on_boundary and near(point[0], 0.0))

bc_left = DirichletBC(W, Constant(0.0), left_boundary)


# =============================================================================
# 5. Initial Gaussian concentration
# =============================================================================

C0 = 1.0
X0 = 0.2
sigma = 0.04

C_initial_expression = Expression(
    "C0*exp(-pow(x[0]-X0, 2)/(2.0*pow(sigma, 2)))",
    degree=4,
    C0=C0,
    X0=X0,
    sigma=sigma
)

C_initial_function = interpolate(C_initial_expression, W)

bc_left.apply(C_initial_function.vector())

# =============================================================================
# 6. Constants independent of beta and Pe
# =============================================================================

epsilon = Constant(epsilon_value)

dt = Constant(dt_value)

theta = Constant(0.5)


# =============================================================================
# 7. Function solving one (beta, Pe) parameter pair
# =============================================================================

def solve_max_concentration(beta_value, Pe_value):

    """
    Solve

        C_T + U_eff C_X
            = (D_eff C_X)_X - U_eff,X C

    for one pair (beta, Pe).

    Return

        max_{0 <= T <= T_final} max_X C(X,T).

    Therefore the returned value is the largest concentration reached
    anywhere in the channel at any point during the simulation.
    """

    beta = Constant(float(beta_value))

    Pe = Constant(float(Pe_value))

    P_X = (-beta*ufl.cosh(beta*(1.0 - X))/ufl.sinh(beta))

    P_XX = (beta**2*ufl.sinh(beta*(1.0 - X))/ufl.sinh(beta))

    P_XXX = (-beta**3* ufl.cosh(beta*(1.0 - X))/ufl.sinh(beta))

    D_s = ((8.0 / 945.0)*P_X**2)

    D_s_X = ((16.0 / 945.0)*P_X*P_XX)

    K_s = ((4.0 / 315.0)*P_X*P_XX - (2.0 / (3.0 * epsilon * Pe))*P_X)

    U_eff = (epsilon**2*Pe**2*(K_s + D_s_X))

    D_eff = (epsilon**2*(1.0 + Pe**2 * D_s))


    # =========================================================================
    # Weak formulation
    # =========================================================================

    spatial_form = (U_eff*C.dx(0)*q*dx + D_eff*C.dx(0)*q.dx(0)*dx + U_eff.dx(0)*C*q*dx)

    # =========================================================================
    # Crank-Nicolson discretisation
    #     (C^{n+1} - C^n)/dt + theta L(C^{n+1}) + (1-theta) L(C^n) = 0
    # =========================================================================

    a_left = ((C / dt)*q*dx + theta*spatial_form)

    a_right = ((C / dt)*q*dx - (1.0 - theta)*spatial_form)

    A_matrix = assemble(a_left)

    B_matrix = assemble(a_right)

    # Apply homogeneous inlet boundary condition.

    bc_left.apply(A_matrix)


    linear_solver = LUSolver(A_matrix)


    # =========================================================================
    # Reset initial concentration
    # =========================================================================

    C_n = Function(W)

    C_n.assign(C_initial_function)

    C_sol = Function(W)


    # =========================================================================
    # Initialise maximum-over-time
    # =========================================================================

    # Include T = 0 in the maximum.
    #
    # Because the initial Gaussian has peak approximately C = 1, maximum_over_time >= 1.

    maximum_over_time = np.max(C_n.vector().get_local())


    # =========================================================================
    # Time stepping
    # =========================================================================

    for n in range(num_steps):

        b = C_n.vector().copy()

        B_matrix.mult(C_n.vector(), b)

        bc_left.apply(b)

        linear_solver.solve(C_sol.vector(), b)


        # =====================================================================
        # Check maximum concentration at this timestep
        # =====================================================================

        current_maximum = np.max(C_sol.vector().get_local())

        # Update the largest concentration that has occurred anywhere
        # during the simulation.

        maximum_over_time = max(maximum_over_time, current_maximum)

        # ---------------------------------------------------------------------
        # Advance solution
        # ---------------------------------------------------------------------

        C_n.assign(C_sol)

    return maximum_over_time


# =============================================================================
# 8. Run beta-Pe parameter sweep
# =============================================================================

# Rows correspond to Pe.
#
# Columns correspond to beta.

max_concentration = np.zeros((len(Pe_values), len(beta_values)))


number_of_simulations = (len(Pe_values)*len(beta_values))

simulation_number = 0


for i, Pe_value in enumerate(Pe_values):

    for j, beta_value in enumerate(beta_values):

        simulation_number += 1

        print(
            f"Simulation "
            f"{simulation_number}/"
            f"{number_of_simulations}: "
            f"beta = {beta_value:.4g}, "
            f"Pe = {Pe_value:.4g}"
        )

        max_concentration[i, j] = (
            solve_max_concentration(
                beta_value,
                Pe_value
            )
        )

        # Print the resulting maximum as well.
        print(
            f"    maximum concentration = "
            f"{max_concentration[i, j]:.8f}"
        )

# =============================================================================
# 9. Save numerical data
# =============================================================================

output_directory = ("maximum_concentration_parameter_sweep")

os.makedirs(
    output_directory,
    exist_ok=True
)

np.save(
    os.path.join(
        output_directory,
        "beta_values.npy"), 
        beta_values
)

np.save(
    os.path.join(
        output_directory,
        "Pe_values.npy"),
    Pe_values
)


np.save(
    os.path.join(
        output_directory,
        "max_concentration_over_time.npy"),
    max_concentration
)


# =============================================================================
# 10. imshow plot
# =============================================================================

plt.figure(figsize=(7, 6))

image = plt.imshow(

    max_concentration,
    origin="lower",
    aspect="auto",
    extent=[
        beta_values[0],
        beta_values[-1],
        Pe_values[0],
        Pe_values[-1]
    ],
    interpolation="nearest",
    cmap="viridis"
)


# =============================================================================
# 11. Colourbar
# =============================================================================

colourbar = plt.colorbar(image)


colourbar.set_label(
    r"$\max_{T,X} C^{(0)}(X,T)$"
)


# =============================================================================
# 12. C_max = 1 contour
# =============================================================================

contour = plt.contour(
    beta_values,
    Pe_values,
    max_concentration,
    levels=[1.0],
    colors="white",
    linewidths=2.5
)


plt.clabel(
    contour,
    fmt={
        1.0: r"$C_{\max}=1$"
    },
    fontsize=14,
    inline=True
)


# =============================================================================
# 13. Axes
# =============================================================================

plt.xlabel(r"$\beta$")

plt.ylabel(r"$\operatorname{Pe}$")


# plt.title(
#     rf"$\max_{{0\leq T\leq {T_final:g}}}"
#     rf"\max_X C^{{(0)}}(X,T)$"
# )

plt.tight_layout()

# =============================================================================
# 14. Save figure
# =============================================================================

figure_path = os.path.join(
    output_directory,
    "maximum_concentration_over_time_imshow.svg"
)


plt.savefig(
    figure_path,
    bbox_inches="tight"
)

plt.show()