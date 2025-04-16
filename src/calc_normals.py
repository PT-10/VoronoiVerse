import open3d as o3d
import numpy as np
from scipy.spatial import Voronoi
import polyscope as ps
from plyfile import PlyData

def compute_voronoi_pca_normals(pcd, k=20):
    # Step 1: Convert to numpy array
    points = np.asarray(pcd.points)
    
    # Step 2: KDTree for neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    normals = []
    confidences = []

    for i in range(len(points)):
        # Get k nearest neighbors
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx]

        # Compute Voronoi diagram
        try:
            vor = Voronoi(neighbors)
        except:
            # Degenerate configuration
            normals.append([0, 0, 0])
            confidences.append(0)
            continue

        # PCA on neighbors
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # Smallest eigenvector

        # Confidence: ratio of smallest eigenvalue to sum
        confidence = 1 - eigvals[0] / (eigvals.sum() + 1e-8)

        normals.append(normal)
        confidences.append(confidence)

    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
    return np.array(normals), np.array(confidences)

# Example usage:
pcd = o3d.io.read_point_cloud("src/data/bunny/reconstruction/bun_zipper.ply")
normals, confidences = compute_voronoi_pca_normals(pcd)

def read_ply(filepath):
    plydata = PlyData.read(filepath)
    vertices = np.vstack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')]).T
    return vertices

points = read_ply("src/data/bunny/reconstruction/bun_zipper.ply")

# ps.init()
# ps_cloud = ps.register_point_cloud("point cloud", points)
# ps_cloud.add_vector_quantity("normals", normals, enabled=True)

# ps.show()
