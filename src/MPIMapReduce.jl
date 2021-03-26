module MPIMapReduce
using MPI

include("concat.jl")

export pmapreduce

function __init__()
    if !MPI.Initialized()
        MPI.Init()
    end
end

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
_reduce(x, op::Union{typeof(vcat), typeof(hcat), Cat}, root, comm) = _gatherv(x, op, root, comm)

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

function _mapreduce_local(f, op, it)
    isempty(it) && return nothing
    mapreduce(x -> f(x...), op, it)
end

function _reduceroot(m, op, comm, root, emptyiter, allactive)
    # If there are more processes than elements in the iterator, then some processes might
    # not be involved in the mapreduce. Split the communicator to exclude these processes
    # from the final reduction.
    comm_reduce = allactive ? comm : MPI.Comm_split(comm, Int(emptyiter), MPI.Comm_rank(comm))

    np = MPI.Comm_size(comm_reduce)
    if 0 <= root < np
        # Carry out the reduction only on processes that contain some elements
        if !emptyiter
            return _reduce(m, op, root, comm_reduce)
        else
            return nothing
        end
    else
        # In this case the iterator on root is empty, so no reduction happens there
        # We carry out the reduction on comm_reduce and transfer the result to root
        s = !emptyiter ? _reduce(m, op, 0, comm_reduce) : nothing
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

"""
    pmapreduce(f, op, iterators...; [root = 0], [comm = MPI.COMM_WORLD])

Apply function `f` to each element(s) in `iterators`, and then reduce the result using the elementwise
binary reduction operator `op`.
Both the `map` and the reduction are evaluated in parallel over the processes corresponding to the communicator `comm`.
The result of the operation is returned at `root`, while `nothing` is returned at the other processes.

The result of `pmapreduce` is identical to that of `mapreduce(f, op, iterators...)`.

The reduction operator `op` may be any Julia operator supported by `MPI`, as well as one of `vcat`, `hcat` and
[`Cat`](@ref).

!!! warn
    `MPI` reduction operators (eg. `MPI.SUM`) are not supported at present.
"""
function pmapreduce(f, op, iterators...; root::Integer = 0, comm::MPI.Comm = MPI.COMM_WORLD)
    if !(0 <= root < MPI.Comm_size(comm))
        throw(ArgumentError("root = $root does not satisfy  0 <= root < $(MPI.Comm_size(comm))"))
    end

    itzip = zip(iterators...)
    Niter = length(itzip)

    if Niter == 0
        throw(ArgumentError("reducing over an empty collection is not allowed"))
    end

    it, np_mapreduce, allactive, emptyiter = _split_iterators(itzip, Niter, comm)

    rank = MPI.Comm_rank(comm)

    if np_mapreduce == 1
        # Either 1 process or 1 element in the iterator
        # no need for MPI here
        if isroot(comm, root)
            return mapreduce(f, op, iterators...)
        else
            return nothing
        end
    end

    m = _mapreduce_local(f, op, it)

    r = _reduceroot(m, op, comm, root, emptyiter, allactive)

    return r
end

end
