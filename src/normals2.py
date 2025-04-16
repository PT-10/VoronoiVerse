import open3d as o3d
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import KDTree

def compute_anisotropy(cov):
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)  # Avoid divide-by-zero
    isotropy = eigvals.min() / eigvals.max()
    return 1.0 - isotropy, eigvals

def compute_covariance(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    return np.dot(centered.T, centered) / len(points)

def compute_voronoi_normals_with_anisotropy(pcd, k=50, threshold=0.9):
    points = np.asarray(pcd.points)
    tree = KDTree(points)

    normals = []
    confidences = []

    for i, p in enumerate(points):
        _, idxs = tree.query(p, k=k)
        max_anisotropy = -1
        best_cov = None

        combined_points = [p]
        for j in idxs[1:]:
            combined_points.append(points[j])
            region_points = np.array(combined_points)

            try:
                vor = Voronoi(region_points)
                cov = compute_covariance(region_points)
            except:
                continue

            anisotropy, eigvals = compute_anisotropy(cov)

            if anisotropy > max_anisotropy:
                max_anisotropy = anisotropy
                best_cov = cov

            if anisotropy >= threshold:
                break

        if best_cov is None:
            normals.append([0, 0, 0])
            confidences.append(0)
            continue

        # Normalize the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(best_cov)
        eigvals = eigvals / eigvals.max()
        norm_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Normal = eigenvector corresponding to smallest eigenvalue
        eigvals_sorted = np.argsort(eigvals)
        normal = eigvecs[:, eigvals_sorted[0]]
        normals.append(normal)
        confidences.append(max_anisotropy)

    normals = np.array(normals)
    confidences = np.array(confidences)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return normals, confidences

# if __name__ == "__main__":
    # import polyscope as ps
    # from plyfile import PlyData
    # pcd = o3d.io.read_point_cloud("src/data/bunny/reconstruction/bun_zipper.ply")

    # normals, confidences = compute_voronoi_normals_with_anisotropy(pcd)

    # def read_ply(filepath):
    #     plydata = PlyData.read(filepath)
    #     vertices = np.vstack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')]).T
    #     return vertices

    # points = read_ply("src/data/bunny/reconstruction/bun_zipper.ply")

    # ps.init()
    # ps_cloud = ps.register_point_cloud("point cloud", points)
    # ps_cloud.add_vector_quantity("normals", normals, enabled=True)

    # ps.show()

