# MPIMapReduce

This package provides a function [`pmapreduce`](@ref) that performs a distributed `mapreduce` operation using MPI.

!!! note
    The `map` operation is performed in batches, therefore the operation is not load-balanced. The iterators are split evenly over the available processes, however this might change in the future.

# Installation

Install the package using

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/jishnub/MPIMapReduce.jl")
```

# Usage

To start using the package, load it and initialize `MPI`.

```julia
using MPIMapReduce

MPI.Init()
```

## Mapreduce

The syntax for a parallel `mapreduce` is similar to that of a serial `mapreduce`, but not exactly the same.
Given a mapping function `f` and a binary elementwise reduction operator `op`, a parallel mapreduce call would look like

```julia
pmapreduce(f, op, iterators...; [root = 0], [comm = MPI.COMM_WORLD])
```

Optionally the root process and the communicator may be specified as the keyword arguments `root` and `comm`. The result of the `mapreduce` operation is returned at the root process, while `nothing` is returned at the other processes.

!!! note
    Unlike `mapreduce`, the operator `op` acts elementwise on the returned values. The operation `pmapreduce(f, op, iterators...)` is therefore equivalent to `mapreduce(f, (x,y) -> op.(x,y), iterators...)`.

## Concatenation

The function [`pmapgatherv`](@ref) may be used to perform a concatenation. The syntax of `pmapgatherv` is similar to `pmapreduce`, except the reduction operator is not applied elementwise. Supported reduction operators are `vcat`, `hcat` and [`Cat`](@ref), where the last operator may be used to perform general concatenations along arbitrary dimensions.

To perform a concatenation, use
```julia
pmapgatherv(f, op, iterators...; [root = 0], [comm = MPI.COMM_WORLD])
```

As in `pmapreduce`, the concatenated result is returned at `root` and `nothing` is returned at the other processes.

# Running scripts

Like most MPI code, scripts using `MPIMapReduce` must be run as

```julia
$ mpiexec -np <np> <julia> <script.jl>
```

where `<np>` is the number of processes, `<julia>` refers to the path to the julia executable, and `<script.jl>` is the julia script that needs to be executed.
