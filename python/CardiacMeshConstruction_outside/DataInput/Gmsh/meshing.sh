#!/bin/bash
FILES=/home/crg17/Desktop/KCL_projects/fitting/python/CardiacMeshConstruction_testing/DataInput/Gmsh/geofiles/*.geo

for f in $FILES
do
  d=${f##*/}
  d=${d%_*_*}
  echo "File: $f, chamber: $d"

  /opt/anaconda3/bin/gmsh "$f" -3 -v 2 -o /home/crg17/Desktop/KCL_projects/fitting/python/CardiacMeshConstruction_testing/DataInput/Gmsh/tetra/"$d"_tetra.vtk

done