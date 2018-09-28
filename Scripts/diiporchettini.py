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

def generateIso(data,iso,mat,frame,state):
    try:
        verts, faces, normals, values = measure.marching_cubes(data, iso)
    except ValueError:
        print('No iso here') 
        return None

    name = 'frame{:03d}_iso_{}_state_{}'.format(frame,iso,state)
    ob = createMeshFromData(name,(0,-30,0),verts,(faces.astype(int)).tolist())
    #ob.scale = [0.09, 0.05, 0.02] # NON SWAPPED
    #             t    g     p
    ob.scale = [0.02, 0.06, 0.06] # SWAPPED
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
    ob.hide_render = True
    ob.keyframe_insert(data_path="hide")
    ob.keyframe_insert(data_path="hide_render")

    bpy.context.scene.frame_set(i)
    ob.hide = False
    ob.hide_render = False
    ob.keyframe_insert(data_path="hide")
    ob.keyframe_insert(data_path="hide_render")
    
    bpy.context.scene.frame_set(i+1)
    ob.hide = True
    ob.hide_render = True
    ob.keyframe_insert(data_path="hide")
    ob.keyframe_insert(data_path="hide_render")
    
    for obj in bpy.data.objects:
        obj.select = False


bpy.ops.object.select_all(action='DESELECT')



#G_E='/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/results/o-newoneWithNACnow_0000/Gaussian00*.h5'

G_Exp = '/home/alessio/Desktop/USETHISINBLENDER_0001/Gaussian*.h5'
G_Exp = '/home/alessio/m-dynamicshere/results/1_2_nac_0001/Gaussian*.h5'

allH5 = sorted(glob.glob(G_Exp))

from quantumpropagator import abs2

for i,fn in enumerate(allH5[:]):
    for state in range(2):
        nameMaterial = "Material.{:03d}".format(state+1)
        mat = bpy.data.materials.get(nameMaterial)
    
        print('doing {}'.format(fn))
    
        wf = openh5(fn,'WF')
        ground2 = abs2(wf[:,:,:,state])
        
        ground = np.swapaxes(ground2,0,2)
        
        for iso in [0.0001, 0.001, 0.003]:
        #for iso in [0.01]:
            generateIso(ground,iso,mat,i,state)

bpy.context.scene.frame_set(1)

def doThis(strei):
    for obj in bpy.data.objects:
        obj.select = False
    for obj in bpy.data.objects:
        object_name = obj.name
        if object_name[:5] == 'frame':
            if strei == 'sele':
                bpy.data.objects[object_name].select = True
            elif strei == 'dele':
                bpy.data.objects[object_name].select = True
                bpy.ops.object.delete()
            else:
                print('I just deselect')    
                

# workaround to REMESH the MC-Mesh
#ob2 = createMeshFromData('dual_contouring',(0,0,0),verts,(faces.astype(int)).tolist())

#bpy.ops.object.modifier_add(type='REMESH')
#bpy.context.object.modifiers["Remesh"].octree_depth = 5
#bpy.context.object.modifiers["Remesh"].mode = 'SMOOTH'