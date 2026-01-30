# How to call:
# from refinement import MeshRefiner
# refined_mesh = MeshRefiner.refine("name.stl", iterations=1, output="name_refined.stl")

import numpy as np
from stl import mesh
import os

class MeshRefiner:
    @staticmethod
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

    @staticmethod
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
            refined_array = np.array(list(map(MeshRefiner.refinement_one_triangle, refined_array)))
            refined_array = np.reshape(refined_array, (n_triangles, 3, 3))
        return refined_array

    @staticmethod
    def refine(input_file, iterations=1, output_file=None):
        """Refine an STL file and save the result."""
        m = mesh.Mesh.from_file(input_file)
        triangles = m.vectors

        refined_triangles = MeshRefiner.refinement_triangulation(triangles, iterations)

        refined_mesh = mesh.Mesh(np.zeros(refined_triangles.shape[0], dtype=mesh.Mesh.dtype))
        refined_mesh.vectors[:] = refined_triangles

        # Create refined .stl file based on name of original .stl file
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_refined.stl"

        refined_mesh.save(output_file)
        print(f"Refined STL saved to {output_file}")
        return refined_mesh