import numpy as np

from core.math3d import *


print("----- Normalize -----")
v = np.array([3.0, 4.0, 0.0])
print(normalize(v))

print()

print("----- Reflection -----")

incident = normalize(np.array([1, -1, 0]))
normal = np.array([0, 1, 0])

print(reflect(incident, normal))

print()

print("----- Basis -----")

ex, ey, ez = build_basis(np.array([1, 2, 3]))

print(ex)
print(ey)
print(ez)

print()

print("Dot Products")

print(dot(ex, ey))
print(dot(ex, ez))
print(dot(ey, ez))