using MPIMapReduce
using MPI

const comm = MPI.COMM_WORLD
const root = 0

y = MPIMapReduce.pmapreduce(x -> x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapreduce(x -> ones(1,2) * x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end

y = MPIMapReduce.pmapreduce(x -> ones(2,2,2) * x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
	show(stdout, MIME"text/plain"(), y)
	println()
end
