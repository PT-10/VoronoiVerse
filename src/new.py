import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.sparse as sp
from plyfile import PlyData
from scipy.spatial import Voronoi

# === Load point cloud and compute Voronoi PCA normals ===
def read_ply(filepath):
    plydata = PlyData.read(filepath)
    vertices = np.vstack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')]).T
    return vertices

def compute_voronoi_pca_normals(pcd, k=20):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    normals = []
    confidences = []
    all_eigvals = []
    all_eigvecs = []

    for i in range(len(points)):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx]

        try:
            vor = Voronoi(neighbors)
        except:
            normals.append([0, 0, 0])
            confidences.append(0)
            all_eigvals.append(np.ones(3) * 1e-4)
            all_eigvecs.append(np.eye(3))
            continue

        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        confidence = 1 - eigvals[0] / (eigvals.sum() + 1e-8)

        normals.append(normal)
        confidences.append(confidence)
        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)

    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
    return np.array(normals), np.array(all_eigvals), np.array(all_eigvecs)


mesh = pv.read("refined_mesh.vtk")

vertices = mesh.points              # (V, 3)
edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]  # (E, 2)
E = len(edges)
V = len(vertices)

import scipy.sparse as sp

def build_incidence_matrix(edges, V):
    E = len(edges)
    rows = np.arange(E)
    v0 = edges[:, 0]
    v1 = edges[:, 1]

    data = np.hstack([-np.ones(E), np.ones(E)])
    row_idx = np.hstack([rows, rows])
    col_idx = np.hstack([v0, v1])

    d0 = sp.coo_matrix((data, (row_idx, col_idx)), shape=(E, V)).tocsr()
    return d0
pcd = o3d.io.read_point_cloud("src/data/bunny/reconstruction/bun_zipper.ply")
normals, all_eigvals, all_eigvecs = compute_voronoi_pca_normals(pcd, k=20)

C_per_vertex = np.zeros((V, 3, 3))
for i in range(V):
    ev = all_eigvals[i]
    evec = all_eigvecs[i]
    ev = ev / (ev.max() + 1e-8)  # Normalize
    C_per_vertex[i] = evec @ np.diag(ev) @ evec.T

def compute_edge_vectors_and_lengths(vertices, edges):
    vecs = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(vecs, axis=1)
    return vecs, lengths

def build_star_1_C(vertices, edges, C_per_vertex):
    e_vecs, lengths = compute_edge_vectors_and_lengths(vertices, edges)
    star = np.zeros(len(edges))
    for i, (v0, v1) in enumerate(edges):
        e = e_vecs[i]
        C_avg = 0.5 * (C_per_vertex[v0] + C_per_vertex[v1])
        numerator = e @ C_avg @ e
        denominator = e @ e
        star[i] = numerator / denominator
    return sp.diags(star)


d0 = build_incidence_matrix(edges, V)
star_1_C = build_star_1_C(vertices, edges, C_per_vertex)

A = d0.T @ star_1_C @ d0

# For B, we can use regular Euclidean approximation
# Placeholder: use identity star operator
star_1 = sp.diags(np.ones(len(edges)))
B = (d0.T @ star_1 @ d0) @ (d0.T @ star_1 @ d0)


print("A shape:", A.shape)
print("B shape:", B.shape)

from scipy.sparse.linalg import eigsh

# Compute largest eigenvalue/eigenvector
eigvals, eigvecs = eigsh(A, k=1, M=B, which='LA')  # Largest λ
F = eigvecs[:, 0]

mesh.point_data['F'] = F
mesh.plot(scalars='F', cmap='coolwarm')

