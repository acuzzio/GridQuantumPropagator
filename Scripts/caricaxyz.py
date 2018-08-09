import bpy
import math
from itertools import combinations
import re

def createMaterial(nameMat,a,b,c):
    bpy.data.materials.new(name=nameMat)
    mat=bpy.data.materials[nameMat]
    mat.diffuse_color[0]=a
    mat.diffuse_color[1]=b
    mat.diffuse_color[2]=c

def addSphere(atomT,index,X,Y,Z,Dim):
    bpy.ops.mesh.primitive_uv_sphere_add(size=Dim, view_align=False, enter_editmode=False, location=(X, Y, Z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    bpy.ops.object.shade_smooth()
    name = atomT + str(index)
    bpy.context.object.name = name
    bpy.context.object.data.materials.append(bpy.data.materials[atomT])
    key(type='Location')

def moveSphere(atomT,index,X,Y,Z):
    name = atomT + str(index)
    bpy.ops.object.select_pattern(pattern=name)
    bpy.context.scene.objects.active = bpy.data.objects[name]
    bpy.context.object.location[0] = X
    bpy.context.object.location[1] = Y
    bpy.context.object.location[2] = Z
    key(type='Location')

def frames(fname):
    f = open(fn)
    atoms=int(f.readline())
    f.close()
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return int((i + 1)/(atoms+2))

def createCyl(rad,dep,X,Y,Z,atomT):
    bpy.ops.mesh.primitive_cylinder_add(radius=rad, depth=dep, view_align=False, enter_editmode=False, location=(X, Y, Z), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
    bpy.ops.object.shade_smooth()
    bpy.context.object.data.materials.append(bpy.data.materials[atomT])

def putCylBetween(atom1,atom2):
    Lab1  = atom1.name
    Lab2  = atom2.name
    Mat1  = Lab1[0]
    Mat2  = Lab2[0]
    # I need to check the distance, so I take the pair name 'OC', transform into 'CO' and check the value
    toCheckDistance=''.join(sorted(Mat1+Mat2))
    bondLengthMax = bondLengths[toCheckDistance] + 0.3
    X1 = atom1.location[0]
    Y1 = atom1.location[1]
    Z1 = atom1.location[2]
    X2 = atom2.location[0]
    Y2 = atom2.location[1]
    Z2 = atom2.location[2]
    X=(X1+X2)/2
    Y=(Y1+Y2)/2
    Z=(Z1+Z2)/2
    dX=(X2-X1)
    dY=(Y2-Y1)
    dZ=(Z2-Z1)
    dim=math.sqrt(dX**2+dY**2+dZ**2)
#    print([bondLengthMax,dim])
    if dim < bondLengthMax:
      if Mat1 == Mat2:
        toghetX=(X1+X2)/2
        toghetY=(Y1+Y2)/2
        toghetZ=(Z1+Z2)/2
        theta=math.acos(dZ/dim)
        phi=math.atan2(dY,dX)
        createCyl(0.1,dim,toghetX,toghetY,toghetZ,Mat1)
#        bpy.context.object.rotation_euler[1] = theta 
#        bpy.context.object.rotation_euler[2] = phi 
        bpy.context.object.name = "Cyl"+ Lab1 + Lab2
#        key(type='LocRotScale')
      else:    
        half=dim/2
        toFirstX=(X1+X)/2
        toFirstY=(Y1+Y)/2
        toFirstZ=(Z1+Z)/2
        toSecondX=(X2+X)/2
        toSecondY=(Y2+Y)/2
        toSecondZ=(Z2+Z)/2
        theta=math.acos(dZ/dim)
        phi=math.atan2(dY,dX)       
        createCyl(0.1,half,toFirstX,toFirstY,toFirstZ,Mat1)
#        bpy.context.object.rotation_euler[1] = theta 
#        bpy.context.object.rotation_euler[2] = phi
        bpy.context.object.name = "Cyl"+ Lab2 + Lab1
#        key(type='LocRotScale')
        createCyl(0.1,half,toSecondX,toSecondY,toSecondZ,Mat2)
#        bpy.context.object.rotation_euler[1] = theta 
#        bpy.context.object.rotation_euler[2] = phi
        bpy.context.object.name = "Cyl"+ Lab1 + Lab2
#        key(type='LocRotScale')

def putPaletti():
    meshes=[]    
    all=bpy.data.objects
    for ii in all:
      if ii.type == 'MESH':
        meshes.append(ii)
    for iii in combinations(meshes,2):
      atom1=iii[0]
      atom2=iii[1]
      putCylBetween(atom1,atom2)
          
def movePaletti():
    cilinders=[]    
    all=bpy.data.objects
    for ii in all:
      if ii.name[0:3] == 'Cyl':
        cilinders.append(ii)
    for iii in cilinders:
      atoms=iii.name[3:]
      splittedSpaced = re.split(r"([A-Z][0-9]*)", atoms)
      splitted  = [x for x in splittedSpaced if x != '']
      atom1n = splitted[0]
      atom2n = splitted[1]
      atom1  = bpy.data.objects[atom1n]
      atom2  = bpy.data.objects[atom2n]
      moveCylinder(atom1,atom2,iii)

def moveCylinder(atom1,atom2,cyl):
    Lab1  = atom1.name
    Lab2  = atom2.name
    Mat1  = Lab1[0]
    Mat2  = Lab2[0]
#     ([atom1,atom2,cyl])
    oldZ = cyl.rotation_euler[2]
    cyl.rotation_euler[0] = 0
    cyl.rotation_euler[1] = 0
    cyl.rotation_euler[2] = 0
    X1 = atom1.location[0]
    Y1 = atom1.location[1]
    Z1 = atom1.location[2]
    X2 = atom2.location[0]
    Y2 = atom2.location[1]
    Z2 = atom2.location[2]
    X=(X1+X2)/2
    Y=(Y1+Y2)/2
    Z=(Z1+Z2)/2
    dX=(X2-X1)
    dY=(Y2-Y1)
    dZ=(Z2-Z1)
    dim=math.sqrt(dX**2+dY**2+dZ**2)
    if Mat1 == Mat2:
        toghetX=(X1+X2)/2
        toghetY=(Y1+Y2)/2
        toghetZ=(Z1+Z2)/2
        cyl.dimensions[2]=dim
        phi = math.atan2(dY, dX) 
        theta = math.acos(dZ/dim) 
        cyl.rotation_euler[1] = theta
        if abs(phi) > 1.0 and phi * oldZ < 0:
           cyl.rotation_euler[2] = phi + (math.pi*2)
        else:
           cyl.rotation_euler[2] = phi 
        cyl.location[0] = toghetX
        cyl.location[1] = toghetY
        cyl.location[2] = toghetZ
        #bpy.ops.object.select_all(action='DESELECT')
        cyl.select = True
        key(type='LocRotScale')
    else:    
        half=dim/2
        toSecX=(X2+X)/2
        toSecY=(Y2+Y)/2
        toSecZ=(Z2+Z)/2
        phi = math.atan2(dY, dX) 
        theta = math.acos(dZ/dim) 
        cyl.rotation_euler[1] = theta
        tau=math.pi*2
        same=phi
        plus=phi+tau
        pluss=phi+tau+tau
        plusss=phi+tau+tau+tau
        minus=phi-tau
        minuss=phi+tau+tau
        minusss=phi+tau+tau+tau
        list=[same,plus,minus,pluss,minuss,plusss,minusss]
        index=min(range(len(list)), key=lambda i: abs(list[i]-oldZ))
        cyl.rotation_euler[2] = list[index]
        cyl.dimensions[2]=half
        cyl.location[0] = toSecX
        cyl.location[1] = toSecY
        cyl.location[2] = toSecZ
        #bpy.ops.object.select_all(action='DESELECT')
        cyl.select = True
        key(type='LocRotScale')

frame = bpy.context.scene.frame_current
key = bpy.ops.anim.keyframe_insert_menu

createMaterial("N",0.0,0.0,1.0)
createMaterial("H",1.0,1.0,1.0)
createMaterial("C",0.3,0.3,0.3)
createMaterial("O",0.8,0.0,0.0)
createMaterial("S",1.0,1.0,0.0)

dimension = {
'H': 0.3,
'C': 0.35,
'O': 0.4,
'N': 0.37,
'S': 0.45
}

bondLengths = {
'HH' : 0.74,
'CH' : 1.09,
'HO' : 0.96,
'HN' : 1.02,
'CC' : 1.54,
'CN' : 1.47,
'CO' : 1.43,
'NN' : 1.45,
'NO' : 1.40,
'OO' : 1.48,
'HS' : 1.34,
'OS' : 1.43,
'CS' : 1.82,
'NS' : 0.50
}

#fn = 'C:/Users/Alessio/Desktop/Dropbox/Blender/punto00_059.md.xyz'
fn = '/home/alessio/Desktop/Dropbox/Blender/punto14_000.md.xyz'


#fn='C:/Users/alessio/Desktop/Dropbox/Blender/water.xyz'
#fn='/home/alessio/Desktop/Dropbox/Blender/water.xyz'
#fn='/home/alessio/Desktop/Dropbox/Blender/geom001.md.xyz'

framesInFile = frames(fn)

f = open(fn)

for i in range(framesInFile):
  bpy.context.scene.frame_current = (i+1)
  atoms=int(f.readline())
  f.readline()
  for j in range(atoms):
    a  = f.readline()
    bb = a.split(" ")
    b  = [x for x in bb if x != '']
    X=float(b[1])
    Y=float(b[3])
    Z=float(b[2])
    atomT = b[0]
    dime = dimension[atomT]
    bpy.ops.object.select_all(action='DESELECT') 
    if i == 0 :
       addSphere(atomT,j,X,Y,Z,dime)      
    else:
       moveSphere(atomT,j,X,Y,Z)
  bpy.ops.object.select_all(action='DESELECT')       
  if i == 0 :
     putPaletti()
     movePaletti()
  else:
     movePaletti()

f.close()

