import numpy as np
from stl import mesh


def refinement_one_triangle(triangle):
    """
    Compute a refinement of one triangle. On every side, the midpoint is added. The three corner points and three
    midpoints result in four smaller triangles.
    :param triangle: array
        array of three points of shape (3, 3) (one triangle)
    :return: array
        array of shape (4, 3, 3) of four triangles
    """
    point1 = triangle[0]
    point2 = triangle[1]
    point3 = triangle[2]
    midpoint12 = (point1 + point2) / 2
    midpoint23 = (point2 + point3) / 2
    midpoint31 = (point3 + point1) / 2
    triangle1 = np.array([point1, midpoint12, midpoint31])
    triangle2 = np.array([point2, midpoint23, midpoint12])
    triangle3 = np.array([point3, midpoint31, midpoint23])
    triangle4 = np.array([midpoint12, midpoint23, midpoint31])
    return np.array([triangle1, triangle2, triangle3, triangle4])


def refinement_triangulation(triangle_array, num_iterations):
    """
    Compute a refinement of a triangulation using the refinement_four_triangles function.
    The number of iteration defines, how often the triangulation has to be refined; n iterations lead to
    4^n times many triangles.
    :param triangle_array: array
        array of shape (num_triangles, 3, 3) of triangles
    :param num_iterations: int
    :return: array
        array of shape (num_triangles*4^num_iterations, 3, 3) of triangles
    """
    refined_array = triangle_array
    for i in range(0, num_iterations):
        n_triangles = refined_array.shape[0] * 4
        refined_array = np.array(list(map(refinement_one_triangle, refined_array)))
        refined_array = np.reshape(refined_array, (n_triangles, 3, 3))
    return refined_array

file_path = r"C:\Users\canca\Downloads\Low-Poly Totodile - 341719\files\totodile.STL"
output_path = r"C:\Users\canca\Downloads\refined model\refined_totodile.stl"

m = mesh.Mesh.from_file(file_path)
triangles = m.vectors

# refine (1 iteration recommended for testing)
refined_triangles = refinement_triangulation(triangles, num_iterations=1)

# save refined mesh
refined_mesh = mesh.Mesh(np.zeros(refined_triangles.shape[0], dtype=mesh.Mesh.dtype))
refined_mesh.vectors[:] = refined_triangles
refined_mesh.save(output_path)

print(f"Refined STL saved to {output_path}")