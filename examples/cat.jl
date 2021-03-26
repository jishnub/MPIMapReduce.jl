using MPIMapReduce
using MPI

const comm = MPI.COMM_WORLD
const root = 0
const redop = MPIMapReduce.Cat([1,2])

y = MPIMapReduce.pmapreduce(x -> x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapreduce(x -> ones(2,3) * x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapreduce(x -> ones(2,3,2) * x^2, redop, 1:3)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end
