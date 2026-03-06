import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import os
import gmsh

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
from dolfinx.io.gmsh import model_to_mesh
# Создаём директорию для VTU файлов
output_dir = "vtu_output"
vtu_dir = os.path.join(output_dir, "vtu_files")
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(vtu_dir, exist_ok=True)





t = 0.0
T = 0.4
num_steps = 300
dt = T / num_steps

for i in range(1):
    gmsh.initialize()
    gmsh.model.add("ring")
    R1 = 0.001   # внутренний радиус
    R2 = 1.0   # внешний радиус
    lc = 0.03   # характерный размер элемента
    outer_circle = gmsh.model.occ.add_circle(0.0, 0.0, 0.0, R2)
    inner_circle = gmsh.model.occ.add_circle(0.0, 0.0, 0.0, R1)
    outer_loop = gmsh.model.occ.add_curve_loop([outer_circle])
    inner_loop = gmsh.model.occ.add_curve_loop([inner_circle])
    ring_surface = gmsh.model.occ.add_plane_surface([outer_loop, inner_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(2, [ring_surface], tag=1, name="Ring surface")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(2)

# Перенос сетки в dolfinx
result = model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2)
domain = result[0]
gmsh.finalize()

V = fem.functionspace(domain, ("Lagrange", 1))
def initial_condition(x, a=5):
    return 0*x[1]
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)
fdim = domain.topology.dim - 1
bcs = []
tol = 1e-3  # Увеличенный допуск
def outer_boundary(x):
    return np.abs(np.sqrt(x[0]**2 + x[1]**2) - R2**2) < tol
def inner_boundary(x):
    return np.abs(np.sqrt(x[0]**2 + x[1]**2) - R1**2) < tol

outer_facets = mesh.locate_entities_boundary(domain, fdim, outer_boundary)
inner_facets = mesh.locate_entities_boundary(domain, fdim, inner_boundary)


if len(outer_facets) > 0:
    bc_outer = fem.dirichletbc(
        PETSc.ScalarType(0),
        fem.locate_dofs_topological(V, fdim, outer_facets),V)
    bcs.append(bc_outer)
if len(inner_facets) > 0:
    bc_inner = fem.dirichletbc(
        PETSc.ScalarType(100),
        fem.locate_dofs_topological(V, fdim, inner_facets),V)
    bcs.append(bc_inner)
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)








def write_vtu(function, time_step, time_value):
    filename = os.path.join(vtu_dir, f"solution_{time_step:04d}.vtu")
    vtk_file = io.VTKFile(domain.comm, filename, "w")
    try:
        vtk_file.write(function, time_value)
    except TypeError:
        try:
            vtk_file.write([function], time_value)
        except TypeError:
            vtk_file.write_function(function, time_value)
    vtk_file.close()
    return filename
if MPI.COMM_WORLD.rank == 0:
    with open(os.path.join(output_dir, "solution.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
write_vtu(uh, 0, t)
if MPI.COMM_WORLD.rank == 0:
    with open(os.path.join(output_dir, "solution.pvd"), "a") as f:
        f.write(f'    <DataSet timestep="{t}" group="" part="0" file="vtu_files/solution_0000.vtu"/>\n')
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter = pyvista.Plotter()
plotter.open_gif(os.path.join(output_dir, "u_time.gif"), fps=10)
grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)
viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)
plotter.add_mesh(
    warped,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[0, 100],
)
plotter.write_frame()
file_counter = 1
for i in range(num_steps):
    t += dt

    # Сборка правой части
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Применяем граничные условия (список bcs)
    apply_lifting(b, [bilinear_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Решение
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Обновление предыдущего шага
    u_n.x.array[:] = uh.x.array

    # Запись каждого 10-го шага
    if i % 10 == 0:
        write_vtu(uh, file_counter, t)
        if MPI.COMM_WORLD.rank == 0:
            with open(os.path.join(output_dir, "solution.pvd"), "a") as f:
                f.write(f'    <DataSet timestep="{t}" group="" part="0" file="vtu_files/solution_{file_counter:04d}.vtu"/>\n')
        PETSc.Sys.Print(f"Записан шаг {i}, время {t:.3f}, файл {file_counter:04d}")
        file_counter += 1

    # Обновление анимации
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()
# Закрытие PVD-файла
if MPI.COMM_WORLD.rank == 0:
    with open(os.path.join(output_dir, "solution.pvd"), "a") as f:
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
plotter.close()
A.destroy()
b.destroy()
solver.destroy()