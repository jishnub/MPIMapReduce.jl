# Examples
## Reduction

We may compute the sum of arrays on each processor as

```julia
using MPIMapReduce
using MPI

y1 = pmapreduce(x -> ones(2) * x, +, 1:5)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    show(stdout, MIME"text/plain"(), y1)
    println()
end
```

This produces the output

```julia
2-element Array{Float64,1}:
 15.0
 15.0
```

## Concatenation

The functions `vcat` and `hcat` may be provided as the reduction operator to carry out concatenations along the first or the second axis respectively.

For example, we may concatenate `Vector`s along the second dimension as
```julia
using MPIMapReduce
using MPI

y = pmapreduce(x -> ones(2) * x^2, hcat, 1:5)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    show(stdout, MIME"text/plain"(), y)
    println()
end
```

This leads to the output
```julia
2×5 Array{Float64,2}:
 1.0  4.0  9.0  16.0  25.0
 1.0  4.0  9.0  16.0  25.0
```

For more general concatenations, use the reduction operator `MPIMapReduce.Cat(dims)`, where `dims` refers to the dimensions along which the concatenation is to be carried out. For example, we may concatenate numbers along the 1st and 2nd dimensions to generate a diagonal matrix as

```julia
using MPIMapReduce
using MPI

y = pmapreduce(x -> x^2, MPIMapReduce.Cat([1,2]), 1:3)

comm = MPI.COMM_WORLD

if MPI.Comm_rank(comm) == 0
    show(stdout, MIME"text/plain"(), y)
    println()
end
```

which leads to the output

```julia
3×3 Array{Int64,2}:
 1  0  0
 0  4  0
 0  0  9
```

!!! note
    Anonymous functions can not be provided as the concatenation operator.
