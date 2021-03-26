# MPIMapReduce

This package provides a function [`pmapreduce`](@ref) that performs a distributed `mapreduce` operation using MPI. The syntax is similar to a standard Julia mapreduce operation, and it should be possible to substitute one with the other to go from a serial to a parallel code.

# Installation

Install the package using

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/jishnub/MPIMapReduce.jl")
```

# Usage

Running

```julia
using MPIMapReduce
```

automatically initializes MPI if it has not beend initialized already.

The syntax for a parallel `mapreduce` is similar to that of a serial `mapreduce`.
Given a mapping function `f` and a binary elementwise reduction operator `op`, a parallel mapreduce call would look like

```julia
using MPIMapReduce

pmapreduce(f, op, iterators...)
```

Optionally the root process and the communicator may be specified as the keyword arguments `root` and `comm`. The result of the `mapreduce` operation is returned at the root process, while `nothing` is returned at the other processes.

The reduction operator may be any Julia binary reduction operator, as well as one of `vcat`, `hcat` and [`MPIMapReduce.Cat`](@ref).

!!! warn
    MPI operators are not supported at present.

# Running scripts

Like most MPI code, scripts using `MPIMapReduce` must be run as

```julia
$ mpiexec -np <np> <julia> <script.jl>
```

where `<np>` is the number of processes, `<julia>` refers to the path to the julia executable, and `<script.jl>` is the julia script that needs to be executed. The path to `mpiexec` might be obtained as

```
julia -e "using MPI; println(MPI.mpiexec_path)"
```
