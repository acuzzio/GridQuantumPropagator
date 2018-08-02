import sys

appen = ['/home/alessio/config/miniconda/envs/quantumpropagator/bin',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python35.zip',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python3.5',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python3.5/plat-linux',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python3.5/lib-dynload',
 '/home/alessio/.local/lib/python3.5/site-packages',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python3.5/site-packages',
 '/home/alessio/Desktop/git/GridQuantumPropagator/src',
 '/home/alessio/config/miniconda/envs/quantumpropagator/lib/python3.5/site-packages/IPython/extensions',
 '/home/alessio/.ipython']

check = '/home/alessio/Desktop/git/GridQuantumPropagator/src'

if check not in sys.path:
    for modumodu in appen:
        sys.path.append(modumodu)

import numpy as np
import bpy
from skimage import measure
#from skimage.draw import ellipsoid

import h5py
import glob


def openh5(fn,dl):
    with h5py.File(fn, 'r') as f5:
        return f5[dl].value


def createMeshFromData(name, origin, verts, faces):
    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new(name, me)
    ob.location = origin
    ob.show_name = True
    # Link object to scene and make active
    scn = bpy.context.scene
    scn.objects.link(ob)
    scn.objects.active = ob
    ob.select = True
    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me.update()    
    return ob

#http://scikit-image.org/docs/dev/auto_examples/plot_marching_cubes.html
# Generate a level set about zero of two identical ellipsoids in 3D
#ellip_base = ellipsoid(6, 10, 16, levelset=True)

def generateIso(data,iso,mat,frame):
    verts, faces, normals, values = measure.marching_cubes(data, iso)
    name = 'frame{}_iso_{}'.format(frame,iso)
    ob = createMeshFromData(name,(0,-30,0),verts,(faces.astype(int)).tolist())
    ob.scale = [0.05, 0.05, 0.05]
    bpy.context.object.location = [0,0,0]
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = 0.03
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)
    
    # enter in and out from editmode : )  <- bug? this will fill the surface better
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()
    
    bpy.context.scene.frame_set(i-1)
    ob.hide = True
    ob.keyframe_insert(data_path="hide")

    bpy.context.scene.frame_set(i)
    ob.hide = False
    ob.keyframe_insert(data_path="hide")
    
    bpy.context.scene.frame_set(i+1)
    ob.hide = True
    ob.keyframe_insert(data_path="hide")
    
    bpy.ops.object.select_all(action='DESELECT')


bpy.ops.object.select_all(action='DESELECT')

mat = bpy.data.materials.get("Material.001")

G_E='/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/results/o-newoneWithNACnow_0000/Gaussian000*.h5'

allH5 = sorted(glob.glob(G_E))

for i,fn in enumerate(allH5):
    
    wf = openh5(fn,'WF')
    ground = wf[:,:,:,0]

    #for iso in [0.001, 0.01, 0.03]:
    for iso in [0.01]:
        generateIso(ground,iso,mat,i)




# workaround to REMESH the MC-Mesh
#ob2 = createMeshFromData('dual_contouring',(0,0,0),verts,(faces.astype(int)).tolist())

#bpy.ops.object.modifier_add(type='REMESH')
#bpy.context.object.modifiers["Remesh"].octree_depth = 5
#bpy.context.object.modifiers["Remesh"].mode = 'SMOOTH'