#!/usr/bin/env bash

date
/home/crg17/Desktop/KCL_projects/fitting/python/Historia/venv/bin/deformetrica compute predef_model_shooting.xml -p optimization_parameters.xml -o output_shooting
rm output_shooting/*ControlPoints*
rm output_shooting/*Momenta*

mv output_shooting/Shooting*tp_10* ../../../DataOutput/Deformetrica/MorphedModels
rm -r output_shooting
cd ../../..
