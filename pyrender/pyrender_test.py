import trimesh
import pyrender
import numpy as np
tm = trimesh.load('wheelchair_new_compressed.obj')
radii = np.linalg.norm(tm.vertices - tm.center_mass, axis=1)
tm.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')
tm.export('wheelchair_new_compressed2.obj')
# tm.visual.vertex_colors = np.random.uniform(size=tm.vertices.shape)
# tm.visual.face_colors = np.random.uniform(size=tm.faces.shape)

# tm = trimesh.creation.uv_sphere()
# tm.vertices *= (np.random.random(3) + 1 ) * 2
# radii = np.linalg.norm(tm.vertices - tm.center_mass, axis=1)
# tm.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')

mesh = pyrender.Mesh.from_trimesh(tm)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

