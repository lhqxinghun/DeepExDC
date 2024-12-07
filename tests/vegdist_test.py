import numpy as np
import pyvegdist

# Create example data
x = np.array([[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0,8.0,9.0]]).copy(order='C')

# Compute distance matrix using different methods
dist_chord = pyvegdist.compute_distance_matrix("chord", x)
dist_canberra = pyvegdist.compute_distance_matrix("canberra", x)
dist_manhattan = pyvegdist.compute_distance_matrix("manhattan", x)
dist_euclidean = pyvegdist.compute_distance_matrix("euclidean", x)

print("Chord Distance Matrix:\n", dist_chord)
print("Canberra Distance Matrix:\n", dist_canberra)
print("Manhattan Distance Matrix:\n", dist_manhattan)
print("Euclidean Distance Matrix:\n", dist_euclidean)