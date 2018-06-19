import numpy as np
import bpy
import re


def makeMaterial(name, diffuse):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    return mat

def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)

#winl = 'C:\\Users\\Alessio\\Desktop\\Dropbox\\Blender\\SurfaceLoader\\'

#a = np.loadtxt('1.txt')
#b = np.loadtxt('2.txt')
#c = np.loadtxt('3.txt')
#a = np.fromfile(winl + 'fullA.txt')
#b = np.fromfile(winl + 'fullB.txt')
#c = np.fromfile(winl + 'fullC.txt')

a = np.fromfile('fullA.txt')
b = np.fromfile('fullB.txt')
c = np.fromfile('fullC.txt')
#d = np.fromfile('fullE.txt')
d = np.fromfile('fullD_x_0.txt')

nstates = 8

ax1N = a.size
ax2N = b.size
ax3N = c.size

d = d.reshape((ax1N,ax2N,ax3N,nstates))

dmin = np.min(d)
d = d - (dmin)
amin = np.min(a)
a = a - amin
bmin = np.min(b)
b = b - bmin
cmin = np.min(c)
c = c - cmin


scaleZ = 30
scaleX = 5


bpy.context.scene.frame_end = ax1N-1
bpy.context.scene.frame_start = 0

for state in range(8):

  verts = []
  edges = []
  faces = []

  for m in range(1):
      for j in range(ax2N):
          for k in range(ax3N):
               value = d[m,j,k,state]*scaleZ
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
    for i in bpy.context.scene.objects:
        nameMesh = i.name
        print(nameMesh,len(i.data.vertices))
        g = 0   # <- a counter for the vertexes
        for j in i.data.vertices:
            state = int(re.findall(r'\d+',nameMesh)[0]) # works only for mashes with Surface0014
            kk = g % ax3N
            jj = ((g-kk)/ax3N)%ax2N
            z = d[ff,jj,kk,state]*scaleZ
            print(ax1N,ax2N,ax3N,g,kk,jj,ff,z)
            j.co.z = z
            j.keyframe_insert('co', index=2, frame=ff)
            g += 1


colorsName = ['Red','Blue','Green','White','Grey','Yellow','Pink','Celest']
colorsId   = [(1,0,0),(0,0,1),(0,1,0),(1,1,1),(0.5,0.5,0.5),(1,1,0),(1,0,1),(0,1,1)]

makeMaterial('Black',(0,0,0))

cou = 0
for i in bpy.context.scene.objects:
    meterialT = makeMaterial(colorsName[cou],colorsId[cou])
    setMaterial(i, meterialT)
    bbb = makeMaterial('Black',(0,0,0))
    setMaterial(i, bbb)
    bpy.context.scene.objects.active = i
    bpy.ops.object.modifier_add(type='WIREFRAME')
    bpy.context.object.modifiers["Wireframe"].material_offset = 1
    bpy.context.object.modifiers["Wireframe"].use_replace = False
    cou += 1

print(ax1N,ax2N,ax3N)

