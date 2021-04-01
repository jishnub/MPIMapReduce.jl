# MPIMapReduce

[![CI](https://github.com/jishnub/MPIMapReduce.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jishnub/MPIMapReduce.jl/actions/workflows/ci.yml)
[![doc:stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/MPIMapReduce.jl/stable)
[![doc:dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/MPIMapReduce.jl/dev)

Exports two functions:

1. `pmapreduce`, that performs a parallel map-reduce.
2. `pmapgatherv`, that performs a parallel map-concatenation.

In both these cases, the collective operation is carried out using MPI.

See the documentation for details of the usage.
