#!/bin/bash -l
#SBATCH --job-name="test"
#SBATCH --time=00:29:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --output="job.out"
#SBATCH --error="job.err"
#SBATCH --constraint=gpu
##SBATCH --switches=1
export MPICH_MPIIO_STATS=1
export MPICH_MPIIO_HINTS_DISPLAY=1
#export MPICH_MPIIO_CB_ALIGN=2
srcdir=$(pwd);
#cd $SCRATCH/0_temp/10_io_perf/stripe32_1m/
cd $SCRATCH/0_temp/10_io_perf/stripe1_1m/
cp $srcdir/a.out ../a.out # Copy to dir above stripe dir, to avoid striping for the executable (long setup time!)!
cp $srcdir/adios2.xml ./ # Copy into stripe dir, as else application will not find it.
srun ../a.out 
