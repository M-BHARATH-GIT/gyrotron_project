import numpy as np

from core.frame import Frame

frame = Frame.from_z_axis(
    origin=[0, 0, 0],
    direction=[1, 2, 3]
)

print(frame)

print("\nRotation Matrix\n")
print(frame.rotation_matrix)

p_local = np.array([2, 3, 4])

p_global = frame.to_global(p_local)

print("\nGlobal point")
print(p_global)

print("\nBack to local")

print(frame.to_local(p_global))