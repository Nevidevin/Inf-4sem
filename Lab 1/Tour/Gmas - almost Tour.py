import gmsh
import sys
gmsh.initialize()
gmsh.model.add("Cilindr")
lc=0.001

gmsh.model.occ.addTorus(0,0,0,5,3,1)
gmsh.model.occ.addTorus(0,0,0,5,1,2)
gmsh.model.occ.cut([(3,1)],[(3,2)],3)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.8)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write("../../t2.msh")
gmsh.write("../../t2.geo_unrolled")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()




