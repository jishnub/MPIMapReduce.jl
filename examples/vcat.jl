using MPIMapReduce
using MPI

MPI.Init()

const comm = MPI.COMM_WORLD
const root = 0

y = MPIMapReduce.pmapgatherv(x -> x^2, vcat, 1:1)
y2 = mapreduce(x -> x^2, vcat, 1:1)

if MPI.Comm_rank(comm) == root
    @info "MPI value"
    show(stdout, MIME"text/plain"(), y)
    println()
    @info "Julia value"
    show(stdout, MIME"text/plain"(), y2)
    println()
end

y = MPIMapReduce.pmapgatherv(x -> x^2, vcat, 1:5)
y2 = mapreduce(x -> x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
    @info "MPI value"
    show(stdout, MIME"text/plain"(), y)
    println()
    @info "Julia value"
	show(stdout, MIME"text/plain"(), y2)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(1,2) * x^2, vcat, 1:5)
y2 = mapreduce(x -> ones(1,2) * x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
    @info "MPI value"
    show(stdout, MIME"text/plain"(), y)
    println()
    @info "Julia value"
	show(stdout, MIME"text/plain"(), y2)
	println()
end

y = MPIMapReduce.pmapgatherv(x -> ones(2,2,2) * x^2, vcat, 1:5)
y2 = mapreduce(x -> ones(2,2,2) * x^2, vcat, 1:5)

if MPI.Comm_rank(comm) == root
    @info "MPI value"
    show(stdout, MIME"text/plain"(), y)
    println()
    @info "Julia value"
	show(stdout, MIME"text/plain"(), y2)
	println()
end
