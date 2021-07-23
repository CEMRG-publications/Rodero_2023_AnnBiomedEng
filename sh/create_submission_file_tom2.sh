#!/bin/bash

clear

template_path="/home/crg17/Desktop/KCL_projects/fitting/submission_files"
template_name=""
time_days=0
time_hours="00"
time_minutes="00"
time_seconds="00"
ncores=64
nnodes=8
type_of_simulation=""
solver_folder="/scratch/crg17/solver_files/resources"
sim_name="/scratch/crg17/simulations/test_unloading"
mesh_path="/scratch/crg17/meshes"
mesh_name="test_mesh"
AT_path="/scratch/crg17/meshes"
AT_name="test_AT.dat"
LV_EDP_mmHg=
LV_EDP_kPa=
RV_EDP_mmHg=
RV_EDP_kPa=
Ao_EDP_mmHg=
Ao_EDP_kPa=
PA_EDP_mmHg=
PA_EDP_kPa=
spring_BC=
guccione_params="a=1.720,b_f=8.0,b_fs=4.0,b_t=3.0,kappa=1000"
neohookean_params="c=8.375,kappa=1000"
AV_resistance=
PV_resistance=
tanh_params=

function Usage(){
    echo "Script to run the simulations with exactly the same parameters as defined in the template files on Tom2."
    echo "Parameters:"
    echo "--sim_name: Name of the job submitted."
    echo "--ncores: Number of cores per node to use. Default and maximum is 64."
    echo "--nnodes: Number of nodes to use. Default and maximum is 8."
    echo "--type_of_simulation: Type of simulation. Options are unloading or contraction."
    echo "-td/--time_days: Days (with one digit) in wall time when the simulation will be killed. Default is 0."
    echo "-th/--time_hours: Hours (with two digits) in wall time when the simulation will be killed. Default is 00."
    echo "-tm/--time_minutes: Hours (with two digits) in wall time when the simulation will be killed. Default is 00."
    echo "-ts/--time_seconds: Seconds (with two digits) in wall time when the simulation will be killed. Default is 00."
    echo "-h/--help: Parameters usage."
}

function Warnings(){

    if [[ $type_of_simulation -ne "unloading" ]]; then
        if [[ $type_of_simulation -ne "contraction" ]]; then
        echo "Invalid simulation type. Options are unloading or contraction."
        exit 1
        fi
    fi

    nchar="${#time_days}"
    if [ $nchar -ne 1 ]; then
    echo "Days in wall time (-td/--time_days) needed with one digit."
    exit 1
    fi

    nchar="${#time_hours}"
    if [ $nchar -ne 2 ]; then
    echo "Hours in wall time (-th/--time_hours) needed with two digits."
    exit 1
    fi

    nchar="${#time_minutes}"
    if [ $nchar -ne 2 ]; then
    echo "Minutes in wall time (-tm/--time_minutes) needed with two digits."
    exit 1
    fi

    nchar="${#time_seconds}"
    if [ $nchar -ne 2 ]; then
    echo "Seconds in wall time (-ts/--time_seconds) needed with two digits."
    exit 1
    fi

    if [ "$time_hours" -ge 24 ]; then
    echo "Hours in wall time should be less than 24."
    exit 1
    fi

    if [ "$time_minutes" -ge 60 ]; then
    echo "Minutes in wall time should be less than 60."
    exit 1
    fi

    if [ "$time_seconds" -ge 60 ]; then
    echo "Seconds in wall time should be less than 60."
    exit 1
    fi

    if [ "$ncores" -gt 64 ]; then
    echo "Number of cores must not be greater than 64."
    exit 1
    fi

    if [ "$nnodes" -gt 8 ]; then
    echo "Number of nodes must not be greater than 8."
    exit 1
    fi

    if [ -z "$template_name" ]; then
      template_name=$type_of_simulation"_template.in"
    fi

    if [ ! -f "$template_path/$template_name" ]; then
    echo "$template_path/$template_name does not exist."
    exit 1
    fi
    
    }

function Substitute_line {
#    cmd="sed -i '$1"s"/.*/$2/' $simulation_path"/"$simulation_name"
#    cmd="sed -i '$1"s"/.*/$2/' $simulation_path"/"$simulation_name"
    cmd="sed -i '$1 s|.*|$2|' $simulation_path'/'$simulation_name"
    eval "$cmd"
}


while [ "$1" != "" ]; do
    case $1 in
       --ncores)  shift
                                ncores=$1
                                ;;
       --nnodes )  shift
                                nnodes=$1
                                ;;
       --type_of_simulation ) shift
                                         type_of_simulation=$1
                                         ;;
      -td | --time_days )      shift
                                time_days=$1
                                ;;
      -th | --time_hours )    shift
                                time_hours=$1
                                ;;
        -tm | --time_minutes )  shift
                                time_minutes=$1
                                ;;
        -ts | --time_seconds )  shift
                                time_seconds=$1
                                ;;
        --solver_folder ) shift
                          solver_folder=$1
                          ;;
        --template_path ) shift
                          template_path=$1
                          ;;
        --template_name ) shift
                          template_name=$1
                          ;;
        --simulation_path ) shift
                          simulation_path=$1
                          ;;
        --sim_name ) shift
                          sim_name=$1
                          ;;
        --mesh_path ) shift
                          mesh_path=$1
                          ;;
        --mesh_name ) shift
                          mesh_name=$1
                          ;;
        --AT_path ) shift
                          AT_path=$1
                          ;;
        --AT_name ) shift
                          AT_name=$1
                          ;;
        --LV_EDP_mmHg ) shift
                          LV_EDP_mmHg=$1
                          ;;
        --LV_EDP_kPa ) shift
                          LV_EDP_kPa=$1
                          ;;
        --RV_EDP_mmHg ) shift
                          RV_EDP_mmHg=$1
                          ;;
        --RV_EDP_kPa ) shift
                          RV_EDP_kPa=$1
                          ;;
        --Ao_EDP_mmHg ) shift
                          Ao_EDP_mmHg=$1
                          ;;
        --Ao_EDP_kPa ) shift
                          Ao_EDP_kPa=$1
                          ;;
        --PA_EDP_mmHg ) shift
                          PA_EDP_mmHg=$1
                          ;;
        --PA_EDP_kPa ) shift
                          PA_EDP_kPa=$1
                          ;;
        --spring_BC ) shift
                          spring_BC=$1
                          ;;
        --guccione_params ) shift
                          guccione_params=$1
                          ;;
        --neohookean_params ) shift
                          neohookean_params=$1
                          ;;
        --AV_resistance ) shift
                          AV_resistance=$1
                          ;;
        --PV_resistance ) shift
                          PV_resistance=$1
                          ;;
        --tanh_params ) shift
                          tanh_params=$1
                          ;;
        -h | --help )           Usage
                                exit
                                ;;
        * )                     echo "Command $1 not found. Use -h or --help for more info."
                                exit 1
    esac
    shift
done


Warnings

simulation_name=$sim_name".in"

# We copy the template to a new file substituting the lines we want
cmd="mkdir -p "$simulation_path

cmd="cp $template_path/$template_name $simulation_path/$simulation_name"
eval "$cmd"

cmd="chmod +w $simulation_path/$simulation_name"
eval "$cmd"


#-------- Header -----------

# Substitute sim_name
line_num=4
line_str="#SBATCH -J "$sim_name

Substitute_line $line_num "$line_str"

# Substitute WALLTIME

line_num=5
line_str="#SBATCH -t $time_days-$time_hours:$time_minutes:$time_seconds"

Substitute_line $line_num "$line_str"

# Substitute --nodes

line_num=6
line_str="#SBATCH --nodes="$nnodes

Substitute_line $line_num "$line_str"

# Substitute --ntasks-per-node

line_num=7
line_str="#SBATCH --ntasks-per-node="$ncores

Substitute_line $line_num "$line_str"

# Substitute NPROC

NPROC=$((ncores*nnodes))

line_num=11
line_str="NPROC="$NPROC

Substitute_line $line_num "$line_str"

#------- Custom parameters

line_num=15
line_str="solver_folder="\"$solver_folder\"

Substitute_line $line_num "$line_str"

line_num=16
line_str="sim_name="\"$sim_name\"

Substitute_line $line_num "$line_str"

line_num=17
line_str="mesh_path="\"$mesh_path\"

Substitute_line $line_num "$line_str"

line_num=18
line_str="mesh_name="\"$mesh_name\"

Substitute_line $line_num "$line_str"

line_num=19
line_str="AT_path="\"$AT_path\"

Substitute_line $line_num "$line_str"

line_num=20
line_str="AT_name="\"$AT_name\"

Substitute_line $line_num "$line_str"

line_num=21
line_str="LV_EDP_mmHg="$LV_EDP_mmHg

Substitute_line $line_num "$line_str"

line_num=22
line_str="LV_EDP_kPa="$LV_EDP_kPa

Substitute_line $line_num "$line_str"

line_num=23
line_str="RV_EDP_mmHg="$RV_EDP_mmHg

Substitute_line $line_num "$line_str"

line_num=24
line_str="RV_EDP_kPa="$RV_EDP_kPa

Substitute_line $line_num "$line_str"

line_num=25
line_str="Ao_EDP_mmHg="$Ao_EDP_mmHg

Substitute_line $line_num "$line_str"

line_num=26
line_str="Ao_EDP_kPa="$Ao_EDP_kPa

Substitute_line $line_num "$line_str"

line_num=27
line_str="PA_EDP_mmHg="$PA_EDP_mmHg

Substitute_line $line_num "$line_str"

line_num=28
line_str="PA_EDP_kPa="$PA_EDP_kPa

Substitute_line $line_num "$line_str"

line_num=29
line_str="spring_BC="$spring_BC

Substitute_line $line_num "$line_str"

line_num=30
line_str="guccione_params="\"$guccione_params\"

Substitute_line $line_num "$line_str"

line_num=31
line_str="neohookean_params="\"$neohookean_params\"

Substitute_line $line_num "$line_str"

line_num=32
line_str="AV_resistance="$AV_resistance

Substitute_line $line_num "$line_str"

line_num=33
line_str="PV_resistance="$PV_resistance

Substitute_line $line_num "$line_str"

line_num=34
line_str="tanh_params="\"$tanh_params\"

Substitute_line $line_num "$line_str"