# VoronoiVerse
EEL7680 - 3D Shape Analysis Course Project



### Key Files and Directories

- **`src/calc_normals.py`**: Script for calculating normals of a 3D mesh.
- **`src/deluanay_refinement.py`**: Implements Delaunay refinement for mesh processing.
- **`src/data/bunny/reconstruction/`**: Contains the `bun_zipper.ply` point cloud file and related data.
- **`src/refined_mesh.vtk`**: Refined 3D mesh file.
- **`src/main.py`**: Main entry point for the project.


## Dependencies

This project uses the following libraries and tools:

- [PyVista](https://docs.pyvista.org/) for 3D visualization.
- PyGAL - Python bindings for [CGAL](https://www.cgal.org/): computational geometry algorithms.
- [Open3D](http://www.open3d.org/) for point cloud processing.
- Meshio

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/VoronoiVerse.git
   cd VoronoiVerse

2. Install dependencies
    ```bash
    pip install pyvista open3d meshio scipy numpy plyfile polyscope