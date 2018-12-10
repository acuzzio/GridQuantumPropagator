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
import re
import pickle
from quantumpropagator import fromLabelsToFloats, labTranformA

scn = bpy.context.scene

if not scn.render.engine == 'CYCLES':
    scn.render.engine = 'CYCLES'

class Material:
    def set_cycles(self):
        scn = bpy.context.scene
        if not scn.render.engine == 'CYCLES':
            scn.render.engine = 'CYCLES'
    def make_material(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nodes = self.mat.node_tree.nodes
        return self.mat
    def link(self, from_node, from_slot_name, to_node, to_slot_name):
        input = to_node.inputs[to_slot_name]
        output = from_node.outputs[from_slot_name]
        self.mat.node_tree.links.new(input, output)
    def makeNode(self, type, name):
        self.node = self.nodes.new(type)
        self.node.name = name
        self.xpos += 200
        self.node.location = self.xpos, self.ypos
        return self.node
    def dump_node(self, node):
        print (node.name)
        print ("Inputs:")
        for n in node.inputs: print ("	", n)
        print ("Outputs:")
        for n in node.outputs: print ("	", n)
    def new_row():
        self.xpos = 0
        self.ypos += 200
    def __init__(self):
        self.xpos = 0
        self.ypos = 0

def create_cycleMaterial(name,diffuse):
    m = Material()
    m.set_cycles()
    # from chapter 1 of [DRM protected book, could not copy author/title]
    lollo = m.make_material(name)
    diffuseBSDF = m.nodes['Diffuse BSDF']
    a,b,c = diffuse
    diffuseBSDF.inputs["Color"].default_value = [a, b, c, 1]
    materialOutput = m.nodes['Material Output']
    transparentBSDF = m.makeNode('ShaderNodeBsdfTransparent', 'Transparent BSDF')
    transparentBSDF.inputs["Color"].default_value = [1, 1, 1, 1]
    mixShader = m.makeNode('ShaderNodeMixShader', 'Mix Shader')
    m.dump_node(mixShader)
    mixShader.inputs['Fac'].default_value = 0.5
    m.link(transparentBSDF, 'BSDF', mixShader, 1)
    m.link(diffuseBSDF, 'BSDF', mixShader, 2)
    m.link(mixShader, 'Shader', materialOutput, 'Surface')
    return lollo

#def makeMaterial(name, diffuse):
#    mat = bpy.data.materials.new(name)
#    mat.diffuse_color = diffuse
#    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)


scaleZ = 1
scaleX = 1

name_data_file = '/home/alessio/n-Propagation/newExtrapolated_gammaExtrExag.pickle'
with open(name_data_file, "rb") as input_file:
    dictionary_data = pickle.load(input_file)

# a = phi   b = gamma   c = theta

phis_ext = labTranformA(dictionary_data['phis'])
gams_ext = labTranformA(dictionary_data['gams'])
thes_ext = labTranformA(dictionary_data['thes'])

a,b,c = fromLabelsToFloats(dictionary_data)

amin = np.min(a)
a = a - amin
bmin = np.min(b)
b = b - bmin
cmin = np.min(c)
c = c - cmin

f = dictionary_data['potCube']
ax1N,ax2N,ax3N,nstates = f.shape

fmin = np.min(f)
f = f - (fmin)

print(f,f.shape)
bpy.context.scene.frame_end = ax1N-1
bpy.context.scene.frame_start = 0

for state in range(8):

  verts = []
  edges = []
  faces = []

  for m in range(1):
      for j in range(ax2N):
          for k in range(ax3N):
               value = f[m,j,k,state]*scaleZ
               verts.append((b[j]*scaleX,c[k], value))
               #print((b[j],c[k], value))
               if (j != 0 and k != 0):
                   #print(j,k)
                   uno = j*ax3N+k
                   due = uno - 1
                   tre = uno - ax3N
                   qua = due - ax3N
                   #print(uno,due,qua,tre)
                   faces.append([uno,due,qua,tre])

      nameMesh = "Suface{:04d}".format(state)
      mesh = bpy.data.meshes.new(name=nameMesh)
      mesh.from_pydata(verts, edges, faces)
      mesh.update()

      profile_object = bpy.data.objects.new(nameMesh, mesh)
      profile_object.data = mesh  # this line is redundant .. it simply overwrites .data

      scene = bpy.context.scene
      scene.objects.link(profile_object)
      profile_object.select = True


for ff in range(ax1N):    
#for ff in range(2):
    for i in bpy.context.scene.objects:
        nameMesh = i.name
        print(nameMesh,len(i.data.vertices))
        g = 0   # <- a counter for the vertexes
        for j in i.data.vertices:
            state = int(re.findall(r'\d+',nameMesh)[0]) # works only for mashes with Surface0014
            kk = g % ax3N
            jj = ((g-kk)/ax3N)%ax2N
            z = f[ff,jj,kk,state]*scaleZ
            print(ax1N,ax2N,ax3N,g,kk,jj,ff,z)
            j.co.z = z
            j.keyframe_insert('co', index=2, frame=ff)
            g += 1


#colorsName = ['Red','Blue','Green','White','Grey','Yellow','Pink','Celest','grey2','grey3']
#colorsId   = [(1,0,0),(0,0,1),(0,1,0),(1,1,1),(0.5,0.5,0.5),(1,1,0),(1,0,1),(0,1,1),(0.25,0.25,0.25),(0.75,0.75,0.75)]

colorsName = ['Blue','Green','Red','Pink1','Cyan','Yellow','Violet','Grey']
colorsId   = [(0,0,1),(0,1,0),(1,0,0),(1,0,0.5),(0,1,1),(1,1,0),(0.3,0,1),(0.15,0.15,0.15)]

#create_cycleMaterial('Black',(0,0,0))

cou = 0
for i in bpy.context.scene.objects:
    meterialT = create_cycleMaterial(colorsName[cou],colorsId[cou])
    setMaterial(i, meterialT)
    bbb = create_cycleMaterial('Black',(0,0,0))
    setMaterial(i, bbb)
    bpy.context.scene.objects.active = i
    bpy.ops.object.modifier_add(type='WIREFRAME')
    bpy.context.object.modifiers["Wireframe"].material_offset = 1
    bpy.context.object.modifiers["Wireframe"].use_replace = False
    bpy.context.object.modifiers["Wireframe"].thickness = 0.0001
    cou += 1

print(ax1N,ax2N,ax3N)

