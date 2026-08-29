# =============================================================================
# Steady leaky-wall model
#
# Numerical solution of
#
#     C_T + U_eff(X) C_X
#         = d/dX [ D_eff(X) C_X ] - U_eff,X(X) C
#
# where
#
#     U_eff = epsilon^2 Pe^2 (K_s + D_s,X)
#
#     D_eff = epsilon^2 (1 + Pe^2 D_s)
#
# and
#
#     K_s = (4/315) P_X P_XX
#           - (2/(3 epsilon Pe)) P_X
#
#     D_s = (8/945) P_X^2.
#
# =============================================================================


# ---------------------------------------------------------------------------
# 0. Import libraries
# ---------------------------------------------------------------------------

from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import os
import ufl


# ---------------------------------------------------------------------------
# Plotting parameters
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

plt.rcParams["axes.formatter.use_mathtext"] = True


# =============================================================================
# 1. Mesh and finite-element space
# =============================================================================

nx = 1000

mesh = IntervalMesh(nx, 0.0, 1.0)

W = FunctionSpace(mesh, "P", 1)

x = SpatialCoordinate(mesh)

X = x[0]


# =============================================================================
# 2. Model parameters
# =============================================================================

epsilon_value = 0.002
Pe_value = 14.0
beta_value = 2.3

epsilon = Constant(epsilon_value)
Pe = Constant(Pe_value)
beta = Constant(beta_value)


# =============================================================================
# 3. Pressure derivatives
#
# P(X) = sinh(beta(1-X)) / sinh(beta)
# =============================================================================

P_X = (-beta * ufl.cosh(beta * (1.0 - X)) / ufl.sinh(beta))

P_XX = (beta**2 * ufl.sinh(beta * (1.0 - X)) / ufl.sinh(beta))

P_XXX = (-beta**3 * ufl.cosh(beta * (1.0 - X)) / ufl.sinh(beta))

# =============================================================================
# 4. Steady transport coefficients
# =============================================================================

D_s = (8.0 / 945.0) * P_X**2

D_s_X = ((16.0 / 945.0)*P_X*P_XX)

K_s = ((4.0 / 315.0) * P_X * P_XX - (2.0 / (3.0 * epsilon * Pe)) * P_X)

# =============================================================================
# 5. Effective transport coefficients
# =============================================================================

U_eff = (epsilon**2 * Pe**2 * (K_s + D_s_X))

D_eff = (epsilon**2* (1.0 + Pe**2 * D_s))

# =============================================================================
# 6. Boundary conditions
#
#     C(0,T) = 0
#
#     C_X(1,T) = 0
#
# The X = 0 condition is imposed strongly.
#
# The X = 1 condition arises naturally from integration by parts of
#
#     d/dX(D_eff C_X).
# =============================================================================

def left_boundary(point, on_boundary):
    return on_boundary and near(point[0], 0.0)

bc_left = DirichletBC(
    W,
    Constant(0.0),
    left_boundary
)

bcs = [bc_left]


# =============================================================================
# 7. Initial concentration
#
# C(X,0) =
#
#     C0 exp[-(X-X0)^2/(2 sigma^2)]
# =============================================================================

C0 = 1.0
X0 = 0.2
sigma = 0.04

C_initial = Expression(
    "C0*exp(-pow(x[0]-X0, 2)/(2.0*pow(sigma, 2)))",
    degree=4,
    C0=C0,
    X0=X0,
    sigma=sigma,
)

C_n = interpolate(C_initial, W)

C_n.rename("concentration", "C")

# Enforce the inlet condition on the initial state.

bc_left.apply(C_n.vector())


# =============================================================================
# 8. Time discretisation
#
# Crank-Nicolson:
#
#     (C^{n+1} - C^n)/dt
#
#       + 1/2 L(C^{n+1})
#       + 1/2 L(C^n)
#
#       = 0
#
# where L(C) = U_{eff}C_X - (D_eff C_X)_X + U_{eff, X} C
# =============================================================================

T_final = 50

num_steps = 7000

dt_value = T_final / num_steps

dt = Constant(dt_value)

theta = Constant(0.5)


# Trial and test functions

C = TrialFunction(W)

q = TestFunction(W)


# =============================================================================
# 9. Weak formulation
#
# Strong equation:
#     C_T + U_eff C_X - (D_eff C_X)_X + U_eff,X C  = 0.
#
#
# Multiply by q and integrate:
#
# integral C_T q dX + integral U_eff C_X q dX + integral D_eff C_X q_X dX + integral U_eff,X C q dX = 0.
# ==============================================================================

spatial_form = (U_eff*C.dx(0)*q*dx + D_eff*C.dx(0)*q.dx(0)*dx + U_eff.dx(0)*C*q*dx)

# =============================================================================
# 10. Crank-Nicolson matrices
# =============================================================================

a_left = ((C / dt)*q*dx + theta*spatial_form)

a_right = ((C / dt)*q*dx - (1.0 - theta)*spatial_form)

# Assemble time-independent matrices once

A_matrix = assemble(a_left)

B_matrix = assemble(a_right)

# Apply homogeneous inlet condition

bc_left.apply(A_matrix)

# Direct solver

linear_solver = LUSolver(A_matrix)


# =============================================================================
# 11. Output directory
# =============================================================================

output_directory = "antidispersion_gaussian_output"

os.makedirs(
    output_directory,
    exist_ok=True
)


vtkfile = File(
    os.path.join(
        output_directory,
        "solution.pvd"
    )
)


vtkfile << (C_n, 0.0)


# Solution at next timestep

C_sol = Function(W)

C_sol.rename(
    "concentration",
    "C"
)


# =============================================================================
# 12. Times at which concentration profiles are saved
# =============================================================================

snapshot_times = [0.0, 10, 20, 30, 40, 50]

snapshot_steps = {
    int(round(snapshot_time / dt_value)): snapshot_time
    for snapshot_time in snapshot_times[1:]
}


# =============================================================================
# 13. Coordinates for plotting
# =============================================================================

x_coordinates = mesh.coordinates()[:, 0]

sort_indices = np.argsort(x_coordinates)

x_sorted = x_coordinates[sort_indices]


# =============================================================================
# 14. Initialise figure
# =============================================================================

plt.figure(figsize=(6, 6))

# Plot T = 0

C_values = (C_n.compute_vertex_values(mesh)[sort_indices])


plt.plot(
    x_sorted,
    C_values,
    linestyle="--",
    label=r"$T=0$",
    linewidth=2.0
)

# =============================================================================
# 15. Time stepping
# =============================================================================
t = 0.0

for n in range(num_steps):

    step_number = n + 1

    t = step_number * dt_value


    # -----------------------------------------------------------------------
    # Construct RHS: b = B C^n
    # -----------------------------------------------------------------------

    b = C_n.vector().copy()

    B_matrix.mult(C_n.vector(), b)

    # Apply homogeneous Dirichlet condition

    bc_left.apply(b)

    # -----------------------------------------------------------------------
    # Solve
    #
    #     A C^{n+1} = b
    # -----------------------------------------------------------------------

    linear_solver.solve(C_sol.vector(), b)

    # -----------------------------------------------------------------------
    # Save selected times
    # -----------------------------------------------------------------------

    if step_number in snapshot_steps:

        snapshot_time = snapshot_steps[step_number]

        vtkfile << (C_sol, snapshot_time)

        C_values = (C_sol.compute_vertex_values(mesh)[sort_indices])

        plt.plot(
            x_sorted,
            C_values,
            label=rf"$T={snapshot_time:g}$",
            linewidth=2.0
        )


    # -----------------------------------------------------------------------
    # Advance solution
    # -----------------------------------------------------------------------

    C_n.assign(C_sol)


# =============================================================================
# 16. Concentration-profile figure
# =============================================================================

plt.xlabel(r"$X$")
plt.ylabel(r"$C^{(0)}(X,T)$")
plt.xlim(0.0,1.0)
plt.ylim(0.0,2.0)
plt.margins(x=0)
# plt.legend(fontsize=14, ncol=2)

# =============================================================================
# 17. Parameter text
# =============================================================================

parameter_text = (
    "(a)"
    "\n"
    rf"$\beta={beta_value:g}$"
    "\n"
    rf"$\mathrm{{Pe}}={Pe_value:g}$"
)


plt.gca().text(
    0.15,
    0.95,
    parameter_text,
    transform=plt.gca().transAxes,
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=18
)


plt.tight_layout()


# =============================================================================
# 18. Save figure
# =============================================================================

figure_path = os.path.join(
    output_directory,
    "concentration_profiles.svg"
)


plt.savefig(
    figure_path,
    bbox_inches="tight"
)


plt.show()