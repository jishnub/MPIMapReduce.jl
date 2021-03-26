"""
    Cat(dims)

Operator that performs a concatenation over the dimensions `dims`.

# Examples

```jldoctest
julia> c1 = MPIMapReduce.Cat(1);

julia> c1([1], [2])
2-element Array{Int64,1}:
 1
 2

julia> c2 = MPIMapReduce.Cat(2);

julia> c2([1], [2])
1×2 Array{Int64,2}:
 1  2
```
"""
struct Cat{D} <: Function
    dims :: D
end
Cat(c::Cat) = c

dims(c::Cat) = c.dims
dims(::typeof(vcat)) = 1
dims(::typeof(hcat)) = 2

(c::Cat)(x...) = cat(x...; dims = dims(c))

_cat_shape(catdims, sizes, lengths) = Base.cat_shape(catdims, (), sizes...)
_cat_shape(catdims, ::Nothing, lengths) = Base.cat_shape(catdims, (), map(Tuple, lengths)...)

function _similar_cat(c::Cat, x, sizes, lengths)
    catdims = Base.dims2cat(dims(c))
    shape = _cat_shape(catdims, sizes, lengths)
    Base.cat_similar(x, eltype(x), shape)
end
_similar_cat(c::Cat, x::Number, sizes, lengths) = _similar_cat(c, [x], sizes, lengths)
