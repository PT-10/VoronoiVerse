import os
from gep import *
from deluanay_refinement import *
import numpy as np
import open3d as o3d
import pyvista as pv
import meshio

pcd = o3d.io.read_point_cloud("src/data/bunny/reconstruction/bun_zipper.ply")

print("Computing Voronoi PCA normals...")
tensor_C = compute_voronoi_pca_normals(pcd)

points = np.asarray(pcd.points)

domain = BBoxDomain(points)


if not os.path.exists("src/data/refined_mesh.vtk"):
    print("Generating mesh...")
    mesh = pygalmesh.generate_mesh(
    domain,
    extra_feature_edges=None,                   # Optional: feature edges (if any)
    bounding_sphere_radius=0.0,                  # Optional: bounding sphere radius
    lloyd=False,                                  # Apply Lloyd's relaxation for mesh refinement
    odt=False,                                   # Optional: edge insertion for optimization
    perturb=False,                                # Apply perturbation to avoid degenerate configurations
    exude=True,                                  # Exude Steiner points to improve mesh quality
    max_edge_size_at_feature_edges = 0.02,          # Maximum edge size at feature edges
    min_facet_angle=25.0,                        # Minimum facet angle in degrees
    max_radius_surface_delaunay_ball=0.1,        # Maximum radius of the surface Delaunay ball
    max_facet_distance = 0.01,                      # Maximum distance between facet circumcenter and Delaunay ball center
    max_circumradius_edge_ratio=2.0,             # Maximum circumradius to edge ratio (for better Delaunay quality)
    max_cell_circumradius=0.07,                   # Maximum circumradius of tetrahedral cells
    exude_time_limit=100.0,                       # Optional: time limit for exuding Steiner points
    exude_sliver_bound=0.1,                      # Optional: sliver-bound limit for exuding
    verbose=True,                                # Verbose output for debugging
    seed=42                                      # Set a seed for reproducibility
    )

    meshio.write("src/data/refined_mesh.vtk", mesh)

else:
    print("Volume mesh already exists, skipping generation.")
    # === Load tetrahedral mesh ===
    mesh = pv.read("src/data/refined_mesh.vtk")

    A, B = compute_A_and_B(mesh, tensor_C)

    print(A.shape, B.shape)
    # f_values, eigenvalues = solve_implicit_function(A, B, k=1)
    F_classical, eigvals_classical = solve_implicit_function2(A, B, k=1)

    isovalue = np.median(F_classical)
    print("Isovalue (median):", isovalue)

    mesh.point_data["f"] = F_classical
    contour = mesh.contour(isosurfaces=[isovalue], scalars="f")
    points_np = np.asarray(pcd.points)
    point_cloud = pv.PolyData(points_np)

    # Plotting both the contour and original point cloud
    plotter = pv.Plotter()
    plotter.add_mesh(contour, color="orange", show_edges=False, opacity=0.7, label="Anisotropic Isocontour")
    plotter.add_points(point_cloud, color="blue", point_size=4.0, render_points_as_spheres=True, label="Original Point Cloud")
    plotter.add_legend()
    plotter.show()
