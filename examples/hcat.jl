using MPIMapReduce
using MPI

MPI.Init()

const comm = MPI.COMM_WORLD
const root = 0

y = MPIMapReduce.pmapgatherv(x -> ones(2) * x^2, hcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(2,2) * x^2, hcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(2,2,2) * x^2, hcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end
