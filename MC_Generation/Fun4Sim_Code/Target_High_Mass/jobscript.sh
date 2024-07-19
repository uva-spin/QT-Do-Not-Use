#!/bin/bash

dir_macros=$(dirname $(readlink -f $BASH_SOURCE))

jobname=$1
njobs=$2
nevents=$3

echo "njobs=$njobs"
echo "nevents=$nevents"

work=/scratch/$USER/MC/$jobname


mkdir -p $work
chmod -R 01755 $work

cd $dir_macros


for (( id=1; id<=$njobs; id++ ))
do  
  mkdir -p $work/$id/
  chmod -R 01755 $work/$id
  cd $work/$id/
  cp $dir_macros/*.C .
  cp $dir_macros/*.cfg .
  cp $dir_macros/*.txt .
  cp $dir_macros/*.slurm .
  sed -i "s/1234/$nevents/" Fun4Sim.C

  echo "submitting job number = $id"
  sbatch grid.slurm
    

done 
