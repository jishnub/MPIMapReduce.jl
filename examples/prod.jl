using MPIMapReduce
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
sz = MPI.Comm_size(comm)

@time y1 = MPIMapReduce.pmapreduce(x -> ones(2, 2) * x, *, 1:4);

if MPI.Comm_rank(comm) == 0
    show(stdout, MIME"text/plain"(), y1)
    println()
end

@time y2 = MPIMapReduce.pmapreduce(x -> ones(2, 2) * x, MPI.PROD, 1:sz);

if MPI.Comm_rank(comm) == 0
    show(stdout, MIME"text/plain"(), y2)
    println()
end
