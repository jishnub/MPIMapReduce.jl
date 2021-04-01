using MPIMapReduce
using MPI

MPI.Init()

const comm = MPI.COMM_WORLD
const root = 0
const redop = Cat([1,2])

y = MPIMapReduce.pmapgatherv(x -> x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(2,3) * x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(2,3,2) * x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end
