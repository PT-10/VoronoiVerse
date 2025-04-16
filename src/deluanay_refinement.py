import pygalmesh
import numpy as np
import open3d as o3d

class BBoxDomain(pygalmesh.DomainBase):
    def __init__(self, points, padding=0.3):
        # Compute AABB
        self.points = points
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        delta = max_bound - min_bound
        # Enlarge bounding box
        self.lower = min_bound - padding * delta
        self.upper = max_bound + padding * delta
        super().__init__()

        # Calculate the bounding sphere's radius
        center = (self.lower + self.upper) / 2
        self.bounding_sphere_radius = np.linalg.norm(center - self.lower)

    def bounding_box(self):
        return self.lower.tolist(), self.upper.tolist()

    def eval(self, x):
        x = np.array(x)
        min_dist = np.min(np.linalg.norm(self.points - x, axis=1))
        if min_dist < 1e-5:
            return 0.0  # treat exact input points as zero level set
        return min_dist - 0.02


    def get_fixed_points(self):
        # Return the fixed points, if necessary for later use in mesh generation
        return self.points[self.fixed_points_indices]

    def get_bounding_sphere_squared_radius(self):
        return self.bounding_sphere_radius ** 2

# Load point cloud
pcd = o3d.io.read_point_cloud("src/data/bunny/reconstruction/bun_zipper.ply")
points = np.asarray(pcd.points)

# Create bounding box domain
domain = BBoxDomain(points)

# Generate mesh with Delaunay refinement
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

# Save the mesh to a .vtk file using meshio
import meshio
meshio.write("src/data/refined_mesh.vtk", mesh)


import numpy as np
import polyscope as ps
from plyfile import PlyData
import pyvista as pv

# Function to read the point cloud from a PLY file
def read_ply(filepath):
    plydata = PlyData.read(filepath)
    vertices = np.vstack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')]).T
    return vertices

# Load the point cloud
points = read_ply("src/data/bunny/reconstruction/bun_zipper.ply")

# Visualize with Polyscope
ps.init()
ps_cloud = ps.register_point_cloud("Point Cloud", points)

# Load and visualize the volume mesh
mesh = pv.read("src/data/refined_mesh.vtk")
ps_mesh = ps.register_volume_mesh(
    "Volume Mesh",
    np.array(mesh.points),
    np.array(mesh.cells_dict[10]),  # Assuming tetrahedral cells (VTK type 10)
)

# Show the visualization
ps.show()