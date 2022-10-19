import Pkg
Pkg.add("MPI")

using MPI
using Printf
MPI.Init()


println(@sprintf("rank is %d", MPI.Comm_rank(MPI.COMM_WORLD)))
