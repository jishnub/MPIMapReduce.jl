using MPIMapReduce
using MPI

comm = MPI.COMM_WORLD
sz = MPI.Comm_size(comm)

y1 = MPIMapReduce.pmapreduce(x -> ones(2) * x, +, 1:5)

if MPI.Comm_rank(comm) == 0
	show(stdout, MIME"text/plain"(), y1)
	println()
end

y2 = MPIMapReduce.pmapreduce(x -> ones(2) * x, (x,y) -> x .= x .+ y, 1:5)

if MPI.Comm_rank(comm) == 0
	show(stdout, MIME"text/plain"(), y2)
	println()
end
