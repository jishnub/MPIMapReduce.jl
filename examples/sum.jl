using MPIMapReduce
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
sz = MPI.Comm_size(comm)

@time y1 = MPIMapReduce.pmapreduce(x -> ones(2, 2) * x, +, 1:20);
y2 = mapreduce(x -> ones(2, 2) * x, +, 1:20);

if MPI.Comm_rank(comm) == 0
    @info "MPI result"
    show(stdout, MIME"text/plain"(), y1)
    println()
    @info "Julia result"
	show(stdout, MIME"text/plain"(), y2)
	println()
end

@time y1 = MPIMapReduce.pmapreduce(x -> ones(2, 2) * x, MPI.SUM, 1:sz);
y2 = mapreduce(x -> ones(2, 2) * x, +, 1:sz);

if MPI.Comm_rank(comm) == 0
    @info "MPI result"
    show(stdout, MIME"text/plain"(), y1)
    println()
    @info "Julia result"
    show(stdout, MIME"text/plain"(), y2)
    println()
end

@time y1 = MPIMapReduce.pmapreduce((x, y) -> ones(2, 2) * (x + y), MPI.SUM, 1:sz, 1:sz);
y2 = mapreduce((x, y) -> ones(2, 2) * (x + y), +, 1:sz, 1:sz);

if MPI.Comm_rank(comm) == 0
    @info "MPI result"
    show(stdout, MIME"text/plain"(), y1)
    println()
    @info "Julia result"
    show(stdout, MIME"text/plain"(), y2)
    println()
end
