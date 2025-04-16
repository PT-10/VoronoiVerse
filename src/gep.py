import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.spatial import Voronoi
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

# === Load point cloud and compute Voronoi PCA normals ===

def compute_voronoi_pca_normals(pcd, k=8):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    for i in range(len(points)):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx]
        try:
            vor = Voronoi(neighbors)
        except:
            normals.append([0, 0, 0])
            continue
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals.append(normal)
    return np.array(normals)


def compute_A_and_B(mesh, tensor_C):
    """
    Compute matrices A (anisotropic Dirichlet energy) and B (biharmonic energy)
    from a tetrahedral mesh and a tensor field C.

    :param mesh: Mesh object (with vertices and edges)
    :param tensor_C: Tensor values at each vertex
    :return: Sparse matrices A, B
    """
    
    vertices = mesh.points
    # Extract tetrahedral cells (or faces, depending on the mesh type)
    edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]  # Extract edge pairs
    E = len(edges)
    V = len(vertices)
    
    # Initialize sparse matrices A and B (in CSC format for efficiency)
    A = sp.lil_matrix((V, V))  # Anisotropic Dirichlet energy matrix
    B = sp.lil_matrix((V, V))  # Biharmonic energy matrix
    
    # Step 1: Construct the vertex-edge incidence matrix d0 (V x E)
    d0 = np.zeros((V, E))
    for i, (v1, v2) in enumerate(edges):
        d0[v1, i] = 1
        d0[v2, i] = -1

    # Step 2: Compute matrix A (anisotropic Dirichlet energy)
    for i in range(E):
        v1, v2 = edges[i]
        edge_vector = vertices[v2] - vertices[v1]
        edge_length = np.linalg.norm(edge_vector)
        C_edge = (tensor_C[v1] + tensor_C[v2]) / 2  # Average tensor at the vertices
        edge_weight = np.dot(C_edge, edge_vector)/ edge_length
        
        # Update matrix A
        A[v1, v1] = A[v1, v1] + edge_weight
        A[v2, v2] = A[v2, v2] + edge_weight
        A[v1, v2] = A[v1, v2] - edge_weight
        A[v2, v1] = A[v2, v1] - edge_weight
    
    # Step 3: Compute matrix B (biharmonic energy)
    d0_transpose = d0.T
    # B = d0_transpose @ d0  # B = (d0^T * d0)
    # Step 3: Compute matrix B (biharmonic energy)
    B = d0 @ d0.T  # Now B will be (V x V), same shape as A
    return A, B


def compute_isotropic_A_and_B(mesh):
    """
    Compute isotropic matrices A (Dirichlet energy) and B (biharmonic energy)
    from a tetrahedral mesh.
    """
    vertices = mesh.points
    edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]
    E = len(edges)
    V = len(vertices)
    
    A = sp.lil_matrix((V, V))
    B = sp.lil_matrix((V, V))
    
    # Construct incidence matrix d0
    d0 = np.zeros((V, E))
    for i, (v1, v2) in enumerate(edges):
        d0[v1, i] = 1
        d0[v2, i] = -1

    # Step 1: Isotropic Dirichlet energy (uniform edge weights)
    for i in range(E):
        v1, v2 = edges[i]
        edge_vector = vertices[v2] - vertices[v1]
        edge_length = np.linalg.norm(edge_vector)
        edge_weight = 1.0 / edge_length  # Inverse length weight (optional)
        
        A[v1, v1] += edge_weight
        A[v2, v2] += edge_weight
        A[v1, v2] -= edge_weight
        A[v2, v1] -= edge_weight

    # Step 2: Biharmonic matrix
    B = d0 @ d0.T
    return A, B


def solve_implicit_function(A, B, k=1):
    """
    Solves AF = λBF using Cholesky and converts to classical EVP.

    Parameters:
        A (csr_matrix): Stiffness matrix.
        B (csr_matrix): Mass matrix (symmetric positive definite).
        k (int): Number of smallest eigenvalues/eigenvectors to compute.

    Returns:
        F (ndarray): Solution vector defining implicit function.
        eigvals (ndarray): Corresponding eigenvalues.
    """
    # Step 1: Cholesky factorization B = LL^T
    
    B_dense = B
    L = cholesky(B_dense, lower=True)

    # Step 2: Transform A -> L^{-1} A L^{-T}
    L_inv = np.linalg.inv(L)
    A_transformed = L_inv @ A @ L_inv.T

    # Step 3: Solve classical eigenvalue problem
    eigvals, G = np.linalg.eigh(A_transformed)

    # Step 4: Recover F = (L^T)^{-1} G
    F = np.linalg.solve(L.T, G)

    return F[:, :k], eigvals[:k]


def solve_eigsh_directly(A, B):
    # Method 2: Using eigsh directly on A and B (sparse matrices)
    A_sparse = csr_matrix(A)  # Convert A to sparse
    B_sparse = csr_matrix(B)  # Convert B to sparse

    eigvals_sparse, eigvecs_sparse = eigsh(A_sparse, k=1, M=B_sparse, which='LM')

    # Extract the eigenvector corresponding to the largest eigenvalue
    F_sparse = eigvecs_sparse[:, 0]

    # Evaluate function at each mesh vertex (input points only, not Steiner)
    f_values = -F_sparse

    return f_values


def solve_implicit_function2(A, B, k=1):
    n = A.shape[0]

    # Ensure B is positive definite (a requirement for Cholesky factorization)
    B = B + n * np.eye(n)  # Adding a diagonal dominance to make B positive definite

    # Method 1: Using Cholesky factorization and classical operator
    # Since B is now a dense matrix, we can directly use it without converting
    L = cholesky(B, lower=True)  # No need to call .toarray() as B is already a dense matrix

    def apply_M(x):
        # x is a vector
        y = solve_triangular(L.T, x, lower=False)
        z = A @ y
        return solve_triangular(L, z, lower=True)


    n = A.shape[0]
    M_op = LinearOperator((n, n), matvec=apply_M)

    # Solve eigenvalue problem using classical operator
    eigvals_classical, eigvecs_classical = eigsh(M_op, k=k, which='LA')  # Largest algebraic eigenvalue
    G_classical = eigvecs_classical[:, 0]

    # Recover original eigenvector F
    F_classical = solve_triangular(L.T, G_classical, lower=False)

    return F_classical, eigvals_classical