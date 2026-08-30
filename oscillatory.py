from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import os
import ufl
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

# ---------------------------------------------------------------------------
# 1. Mesh and finite-element space
# ---------------------------------------------------------------------------
nx = 1000
mesh = IntervalMesh(nx, 0.0, 1.0)
W = FunctionSpace(mesh, "P", 1)
x = SpatialCoordinate(mesh)
X = x[0]


# ---------------------------------------------------------------------------
# 2. Model parameters
# ---------------------------------------------------------------------------
epsilon_value = 0.002
Pe_value = 14.0
beta_value = 3
Omega_value = 10

epsilon = Constant(epsilon_value)
Pe = Constant(Pe_value)
beta = Constant(beta_value)

# Frequency-dependent parameter
a = np.sqrt(Omega_value / 2.0)


# ---------------------------------------------------------------------------
# 3. Frequency-dependent functions S(a) and T(a)
# ---------------------------------------------------------------------------
# These depend only on Omega, not on X, so evaluate them numerically.
denominator = np.cos(2.0 * a) - np.cosh(2.0 * a)

S_value = (a*(np.sin(2.0 * a) + np.sinh(2.0 * a))/ denominator)

T_value = ((np.sin(2.0 * a) - np.sinh(2.0 * a))/(2.0 * a * denominator))

S_coefficient = Constant(float(S_value))
T_coefficient = Constant(float(T_value))


# ---------------------------------------------------------------------------
# 4. Pressure derivatives
# ---------------------------------------------------------------------------
# P_hat(X) = sinh(beta*(1-X))/sinh(beta)
#
# Therefore:
#
# P_X   = -beta*cosh(beta*(1-X))/sinh(beta)
# P_XX  =  beta^2*sinh(beta*(1-X))/sinh(beta)
# P_XXX = -beta^3*cosh(beta*(1-X))/sinh(beta)
#
# These are algebraically identical to the expressions involving
# 1 - cosh(2*beta) given in the derivation.

P_X = (-beta* ufl.cosh(beta * (1.0 - X))/ufl.sinh(beta))

P_XX = (beta**2*ufl.sinh(beta * (1.0 - X))/ufl.sinh(beta))

P_XXX = (-beta**3*ufl.cosh(beta * (1.0 - X))/ufl.sinh(beta))

# ---------------------------------------------------------------------------
# 5. Oscillatory macrotransport coefficients
# ---------------------------------------------------------------------------
K_o = (P_X * P_XX/ (4.0 * a**4)* ( 4.0 * T_coefficient- (2.0 / 3.0) * S_coefficient - 2.0))

D_o = (P_X**2/ (2.0 * a**4)* (1.0 / 3.0- T_coefficient))

A_o = ((1.0 + S_coefficient)/(6.0 * a**4) * (P_X * P_XXX + P_XX**2))

D_o_X = ((2*P_X*P_XX )/(2.0 * a**4)) * (1.0 / 3.0- T_coefficient)

# ---------------------------------------------------------------------------
# 6. Effective transport coefficients 
# ---------------------------------------------------------------------------


U_eff = epsilon**2 * Pe**2 * (K_o + D_o_X)

D_eff = epsilon**2 * (1.0 + Pe**2 * D_o)


# ---------------------------------------------------------------------------
# 7. Boundary conditions
# ---------------------------------------------------------------------------
# Inlet:
#
#     C(0,T) = 0.
#
# No Dirichlet condition is imposed at X=1. The weak form therefore gives
# the natural condition
#
#     D_eff*C_X = 0,
#
# and, because D_eff > 0, this is equivalent to C_X(1,T) = 0.

def left_boundary(point, on_boundary):
    return on_boundary and near(point[0], 0.0)


bc_left = DirichletBC(W, Constant(0.0), left_boundary)
bcs = [bc_left]


# ---------------------------------------------------------------------------
# 8. Initial Gaussian concentration
# ---------------------------------------------------------------------------
C0 = 1.0
X0 = 0.4
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

bc_left.apply(C_n.vector())


# ---------------------------------------------------------------------------
# 9. Crank-Nicolson time discretisation
# ---------------------------------------------------------------------------
T_final = 1500.0

# dt = 0.1
dt_value = 0.1
num_steps = int(round(T_final / dt_value))

dt = Constant(dt_value)
theta = Constant(0.5)

C = TrialFunction(W)
phi = TestFunction(W)


# ---------------------------------------------------------------------------
# 10. Weak formulation
# ---------------------------------------------------------------------------

spatial_form = (U_eff*C.dx(0)*phi*dx + D_eff*C.dx(0)*phi.dx(0)*dx + U_eff.dx(0)*C*phi*dx)

# Crank-Nicolson:
#
# M(C^{n+1} - C^n)/dt
# + theta*A(C^{n+1})
# + (1-theta)*A(C^n)
# = 0.

a_left = ((C / dt) * phi * dx + theta * spatial_form)

a_right = ((C / dt) * phi * dx - (1.0 - theta) * spatial_form)

A_matrix = assemble(a_left)
B_matrix = assemble(a_right)

bc_left.apply(A_matrix)

linear_solver = LUSolver(A_matrix)


# ---------------------------------------------------------------------------
# 11. Output setup
# ---------------------------------------------------------------------------
output_directory = "oscillatory_macrotransport_output"
os.makedirs(output_directory, exist_ok=True)

vtkfile = File(os.path.join(output_directory, "solution.pvd"))
vtkfile << (C_n, 0.0)

C_sol = Function(W)
C_sol.rename("concentration", "C")


# ---------------------------------------------------------------------------
# 12. Requested snapshot times
# ---------------------------------------------------------------------------
snapshot_times = [
    0.0,
    100.0,
    200.0, 
    400.0, 
    600.0, 
    1000.0,
]

snapshot_steps = {
    int(round(snapshot_time / dt_value)): snapshot_time
    for snapshot_time in snapshot_times[1:]
}

x_coordinates = mesh.coordinates()[:, 0]
sort_indices = np.argsort(x_coordinates)
x_sorted = x_coordinates[sort_indices]


# ---------------------------------------------------------------------------
# 13. Plot the initial concentration
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 6))

C_values = C_n.compute_vertex_values(mesh)[sort_indices]

plt.plot(
    x_sorted,
    C_values,
    linestyle="--",
    linewidth=2.0,
    label=r"$T=0$",
)


# ---------------------------------------------------------------------------
# 14. Time-stepping loop
# ---------------------------------------------------------------------------
saved_frames = 1

for n in range(num_steps):
    step_number = n + 1
    current_time = step_number * dt_value

    # Form the right-hand side b = B*C^n.
    b = C_n.vector().copy()
    B_matrix.mult(C_n.vector(), b)

    # Apply C(0,T)=0.
    bc_left.apply(b)

    # Solve for C^{n+1}.
    linear_solver.solve(C_sol.vector(), b)

    if step_number in snapshot_steps:
        snapshot_time = snapshot_steps[step_number]

        vtkfile << (C_sol, snapshot_time)
        saved_frames += 1

        C_values = C_sol.compute_vertex_values(mesh)[sort_indices]

        plt.plot(
            x_sorted,
            C_values,
            linewidth=1.7,
            label=fr"$T={snapshot_time:g}$",
        )

    C_n.assign(C_sol)


# ---------------------------------------------------------------------------
# 15. Concentration-profile figure
# ---------------------------------------------------------------------------
plt.title(
    "Purely oscillatory macrotransport\n"
    fr"($\beta={beta_value:g}$, $\Omega={Omega_value:g}$)",
    pad=10,
)
plt.xlabel(r"$X$")
plt.ylabel(r"$C^{(0)}(X,T)$")
#plt.legend(fontsize=14, ncol=2)
plt.tight_layout()

figure_path = os.path.join(
    output_directory,
    "concentration_profiles.svg",
)

plt.savefig(figure_path, dpi=300)
plt.show()


# ---------------------------------------------------------------------------
# 16. Diagnostics
# ---------------------------------------------------------------------------
print(f"a = {a:.8g}")
print(f"S(a) = {S_value:.8g}")
print(f"T(a) = {T_value:.8g}")
print(f"Time step = {dt_value:.8g}")
print(f"Simulation complete. Saved {saved_frames} VTK states.")
print(f"Concentration plot saved to: {figure_path}")