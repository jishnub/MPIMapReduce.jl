module MPIMapReduce
using MPI

include("concat.jl")

export pmapreduce
export pmapgatherv
export Cat

abstract type Operation end
struct Reduce <: Operation end
struct Concat <: Operation end

isroot(comm::MPI.Comm, root::Integer) = MPI.Comm_rank(comm) == root

function nelementsdroptake(len, np, rank)
    0 <= rank <= np - 1 || throw(ArgumentError("rank = $rank does not satisfy 0 <= rank <= $(np - 1)"))
    d, r = divrem(len, np)
    drop = d*rank + min(r, rank)
    lastind = d*(rank + 1) + min(r, rank + 1)
    take = lastind - drop
    drop, take
end

_reduce(x, op, root, comm) = MPI.Reduce(x, op, root, comm)

function _vbuffer(x, counts, sizes, c::Cat)
    rcvbuf = _similar_cat(c, x, sizes, counts)
    MPI.VBuffer(rcvbuf, counts)
end

# Return the sizes of the arrays on each worker
# These may differ, so we do this in two steps
# At the first step fetch the ndims from all workers to estimate the number of values to read
# Given this, read the correct number of values from each worker

# Treat numbers like 1-elements arrays in Gather
_sizevec(x::AbstractArray) = collect(size(x))
_sizevec(x::Number) = 1
_ndims(x::AbstractArray) = ndims(x)
_ndims(x::Number) = 1

function _sizes(x, root, comm)
    np = MPI.Comm_size(comm)

    dims = MPI.Gather(_ndims(x), root, comm)
    size_local = _sizevec(x)

    sz = if isroot(comm, root)
        vbuf =  _vbuffer(size_local, dims, nothing, Cat(Val(1)))
        MPI.Gatherv!(size_local, vbuf, root, comm)
    else
        MPI.Gatherv!(size_local, nothing, root, comm)
    end

    if isroot(comm, root)
        sizes = Vector{Vector{Int}}(undef, np)
        offset = 0
        for (ind, nd) in enumerate(dims)
            sizes[ind] = sz[offset .+ (1:nd)]
            offset += nd
        end
        return map(Tuple, sizes)
    end
    return nothing
end

function __reshapedviews(xg, offset, sz)
    inds = offset .+ (1:prod(sz))
    v = @view xg[inds]
    reshape(v, sz...)
end
function _reshapedviews(xg, sizes, counts)
    offsets = cumsum(reduce(vcat, [[zero(eltype(counts))], counts[1:end-1]]), dims = 1)
    [__reshapedviews(xg, offset, sz) for (offset, sz) in zip(offsets, sizes)]
end

# _checkcontiguity(sizes, ::Val{N}) where {N} = _checkcontiguity(sizes, N)
_checkcontiguity(sizes, dims::Integer) = all(x -> length(x) <= dims, sizes)
_checkcontiguity(sizes, dims) = false

_gatherv(x, op, root, comm) = __gatherv(x, op, root, comm, dims(op))

function __gatherv(x, op, root, comm, dims)
    sizes = _sizes(x, root, comm)
    np = MPI.Comm_size(comm)

    if isroot(comm, root)
        counts = map(prod, sizes)
        vbuf =  _vbuffer(x, counts, sizes, Cat(dims))
        xg = MPI.Gatherv!(x, vbuf, root, comm)

        contiguous = _checkcontiguity(sizes, dims)
        if !contiguous
            X = _reshapedviews(xg, sizes, counts)
            xg = reduce(op, X)
        end
        xg
    else
        MPI.Gatherv!(x, nothing, root, comm)
    end
end

function _split_iterators(itzip, Niter, comm)
    rank = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)

    drop, take = nelementsdroptake(Niter, np, rank)
    # split the iterators into parts
    it_local = Iterators.take(Iterators.drop(itzip, drop), take)

    np_mapreduce = min(Niter, np)
    emptyiter = isempty(it_local)

    it_local, np_mapreduce, np == np_mapreduce, emptyiter
end

struct Elementwise{F}
    f :: F
end

(o::Elementwise)(x, y) = o.f.(x, y)
(o::Elementwise)(x::Array, y) = x .= o.f.(x, y)
(o::Elementwise)(x, y::Array) = y .= o.f.(x, y)
(o::Elementwise)(x::Array, y::Array) = x .= o.f.(x, y)

_mapreduce_zipsection(f, op, it) = mapreduce(f, Elementwise(op), it)
function _mapreduce_zipsection(f, op::MPI.Op, it)
    m = map(f, it)
    length(m) == 1 || throw(ArgumentError("more than one value returned"))
    _reduce(first(m), op, 0, MPI.COMM_SELF)
end

function _mapreduce_local_zipsection(::Reduce, f, op, it)
    isempty(it) && return nothing
    _mapreduce_zipsection(x -> f(x...), op, it)
end
function _mapreduce_local_zipsection(::Concat, f, op, it)
    isempty(it) && return nothing
    mapreduce(x -> f(x...), op, it)
end

function _collectroot(reducefn, m, op, comm, root, emptyiter, allactive)
    # If there are more processes than elements in the iterator, then some processes might
    # not be involved in the mapreduce. Split the communicator to exclude these processes
    # from the final reduction.
    comm_reduce = allactive ? comm : MPI.Comm_split(comm, Int(emptyiter), MPI.Comm_rank(comm))

    np = MPI.Comm_size(comm_reduce)
    if 0 <= root < np
        # Carry out the reduction only on processes that contain some elements
        if !emptyiter
            return reducefn(m, op, root, comm_reduce)
        else
            return nothing
        end
    else
        # In this case the iterator on root is empty, so no reduction happens there
        # We carry out the reduction on comm_reduce and transfer the result to root
        s = !emptyiter ? reducefn(m, op, 0, comm_reduce) : nothing
        tag = 32 # arbitrary
        if MPI.Comm_rank(comm) == 0
            MPI.send(s, root, tag, comm)
            return nothing
        elseif isroot(comm, root)
            r, status = MPI.recv(0, tag, comm)
            if status.source == 0 && status.tag == tag
                return r
            else
                error("unable to send the result to root")
            end
        end
    end
    return nothing
end

function _pmapreduce_local(f, op, iterators...; root::Integer = 0, comm::MPI.Comm = MPI.COMM_WORLD)
    if !(0 <= root < MPI.Comm_size(comm))
        throw(ArgumentError("root = $root does not satisfy  0 <= root < $(MPI.Comm_size(comm))"))
    end

    itzip = zip(iterators...)
    Niter = length(itzip)

    if Niter == 0
        throw(ArgumentError("reducing over an empty collection is not allowed"))
    end

    _split_iterators(itzip, Niter, comm)
end

# As such __mapreduce_singleprocess is almost identical to _mapreduce_local_zipsection, but we avoid using the zipped iterator
# This is because mapreduce(f, hcat, itrs...) and reduce(hcat, map(f, itrs...)) differ for single element arrays
# See issue https://github.com/JuliaLang/julia/issues/37917
__mapreduce_singleprocess(::Reduce, f, op, iterators...) = _mapreduce_zipsection(x -> f(x...), op, zip(iterators...))
__mapreduce_singleprocess(::Concat, f, op, iterators...) = mapreduce(f, op, iterators...)

function _mapreduce_singleprocess(m::Operation, root, comm, np_mapreduce, f, op, iterators...)
    # Either 1 process or 1 element in the iterator
    if isroot(comm, root)
        return __mapreduce_singleprocess(m, f, op, iterators...)
    else
        return nothing
    end
end

"""
    pmapreduce(f, op, iterators...; [root = 0], [comm = MPI.COMM_WORLD])

Apply function `f` to each element(s) in `iterators`, and then reduce the result using the elementwise
binary reduction operator `op`.
Both the `map` and the reduction are evaluated in parallel over the processes corresponding to the communicator `comm`.
The result of the operation is returned at `root`, while `nothing` is returned at the other processes.

The result of `pmapreduce` is equivalent to that of `mapreduce(f, (x, y) -> op.(x, y), iterators...)`.

!!! note
    Unlike the standard `mapreduce` operation in Julia, this operation is performed elementwise on the arrays returned
    from the various processes. The returned value must be compatible with the elementwise operation.

!!! note
    MPI reduction operators require each worker to return only one array with an eltype `T` that
    satisfies `isbitstype(T) == true`. Such a limitation does not exist for Julia operators.
"""
function pmapreduce(f, op, iterators...; root::Integer = 0, comm::MPI.Comm = MPI.COMM_WORLD)
    it, np_mapreduce, allactive, emptyiter  = _pmapreduce_local(f, op, iterators...; root = root, comm = comm)
    if np_mapreduce == 1
        return _mapreduce_singleprocess(Reduce(), root, comm, np_mapreduce, f, op, iterators...)
    end
    m = _mapreduce_local_zipsection(Reduce(), f, op, it)
    _collectroot(_reduce, m, op, comm, root, emptyiter, allactive)
end

"""
    pmapgatherv(f, op, iterators...; [root = 0], [comm = MPI.COMM_WORLD])

Apply function `f` to each element(s) in `iterators`, and then reduce the result using the concatenation operator `op`.
Both the `map` and the reduction are evaluated in parallel over the processes corresponding to the communicator `comm`.
The result of the operation is returned at `root`, while `nothing` is returned at the other processes.

The result of `pmapgatherv` is equivalent to that of `mapreduce(f, op, iterators...)`.

The concatenation operator `op` may be one of `vcat`, `hcat` and [`Cat`](@ref).

!!! warn
    Anonymous functions are not supported as the concatenation operator.
"""
function pmapgatherv(f, op, iterators...; root::Integer = 0, comm::MPI.Comm = MPI.COMM_WORLD)
    it, np_mapreduce, allactive, emptyiter  = _pmapreduce_local(f, op, iterators...; root = root, comm = comm)
    if np_mapreduce == 1
        return _mapreduce_singleprocess(Concat(), root, comm, np_mapreduce, f, op, iterators...)
    end
    m = _mapreduce_local_zipsection(Concat(), f, op, it)
    _collectroot(_gatherv, m, op, comm, root, emptyiter, allactive)
end

end
