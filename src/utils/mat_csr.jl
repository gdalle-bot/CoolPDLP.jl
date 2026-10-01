"""
    GPUSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCSR{
        T <: Number,
        Ti <: Number,  # not `<: Integer`: traced indices are `TracedRNumber`s
        V <: DenseVector{T},
        Vi <: DenseVector{Ti},
    } <: AbstractSparseMatrix{T, Ti}
    m::Int
    n::Int
    rowptr::Vi
    colval::Vi
    nzval::V
end

Base.size(A::GPUSparseMatrixCSR) = (A.m, A.n)

SparseArrays.nnz(A::GPUSparseMatrixCSR) = length(A.nzval)
SparseArrays.nonzeros(A::GPUSparseMatrixCSR) = A.nzval

function Base.getindex(
        A::GPUSparseMatrixCSR{T, Ti}, i::Integer, j::Integer
    ) where {T, Ti}
    (; rowptr, colval, nzval) = A
    k1 = rowptr[i]
    k2 = rowptr[i + 1] - 1
    if k1 > k2
        return zero(T)
    else
        k = k1 + searchsortedfirst(view(colval, k1:k2), j) - 1
        if k > k2 || colval[k] != j
            return zero(T)
        else
            return nzval[k]
        end
    end
end

function KernelAbstractions.get_backend(A::GPUSparseMatrixCSR)
    return common_backend(A.rowptr, A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::GPUSparseMatrixCSR)
    return GPUSparseMatrixCSR(
        A.m,
        A.n,
        adapt(to, A.rowptr),
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function GPUSparseMatrixCSR(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    At_csc = SparseMatrixCSC(transpose(A))
    return GPUSparseMatrixCSR(At_csc.n, At_csc.m, At_csc.colptr, At_csc.rowval, At_csc.nzval)
end

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCSR)
    At_csc = SparseMatrixCSC(A.n, A.m, Vector(A.rowptr), Vector(A.colval), Vector(A.nzval))
    return SparseMatrixCSC(transpose(At_csc))
end

function sametype_transpose(A::GPUSparseMatrixCSR)
    A_csc = SparseMatrixCSC(A)
    return adapt(
        get_backend(A),
        GPUSparseMatrixCSR(A_csc.n, A_csc.m, A_csc.colptr, A_csc.rowval, A_csc.nzval)
    )
end

"""
    CSR_WORKGROUP

Workgroup size used by the cooperative CSR kernels. Fixed rather than left to the backend's
occupancy heuristic because [`spmv_csr_vector!`](@ref) sizes its local memory from it.
"""
const CSR_WORKGROUP = 256

"""
    MAX_SUBGROUP

Largest number of work items [`subgroup_size`](@ref) will put on a single row.
"""
const MAX_SUBGROUP = 32

"""
    BATCH_SHIFT

How many powers of two the batch has to grow by before [`subgroup_size`](@ref) halves the
sub-group.
"""
const BATCH_SHIFT = 5

"""
    MIN_BATCHED

Narrowest sub-group [`subgroup_size(A, nbatch)`](@ref) will narrow down to. It is a bound on
the narrowing, not a floor on the result: a matrix whose rows already ask for fewer work
items than this keeps what [`subgroup_size(A)`](@ref) gave it.
"""
const MIN_BATCHED = 4

"""
    MAX_BATCHED

Widest sub-group [`subgroup_size(A, nbatch)`](@ref) will narrow at all. A matrix that wants
more lanes than this wants them because its rows are long, which batching does not change.
"""
const MAX_BATCHED = 16

"""
    subgroup_size(A)

Number of work items to assign to each row of `A`: the largest power of two no greater than
the average number of nonzeros per row, capped at [`MAX_SUBGROUP`](@ref).

One work item per row is the obvious way to write an SpMV, but it makes a row's cost
proportional to its length, so a single long row stalls the whole kernel while the rest of
the device idles. Real LP constraint matrices are strongly row-skewed -- a budget or
knapsack row touching every variable next to rows touching three -- and that cost the
one-work-item-per-row kernel a factor of 40-60 against cuSPARSE on the larger MIPLIB 2017
instances (`square41`, whose longest row holds 17,575 nonzeros against a mean of 338, ran
in 12.3ms where cuSPARSE took 0.2ms). Splitting each row across a sub-group both shortens
the longest row's critical path and makes the lanes read consecutive `nzval`/`colval`
entries.

Sizing the sub-group from the *average* row rather than the longest one is what keeps
matrices with short rows -- where a wide sub-group would leave most lanes idle -- from
regressing: on an L40S this rule was worth a geometric mean of 5.5x across a corpus of
random and MIPLIB matrices, with a worst case of 1.03x, where selecting on the longest row
instead lost 5x on a matrix averaging three nonzeros per row.
"""
function subgroup_size(A::GPUSparseMatrixCSR)
    m = size(A, 1)
    m == 0 && return 1
    return min(prevpow(2, max(nnz(A) ÷ m, 1)), MAX_SUBGROUP)
end

"""
    subgroup_size(A, nbatch)

Number of work items per row when the same `A` is multiplied by `nbatch` right-hand sides
at once: [`subgroup_size(A)`](@ref), narrowed as the batch grows.

A sub-group does two jobs at once and only one of them survives batching. It shortens the
longest row's critical path and makes lanes read consecutive nonzeros, neither of which a
batch changes; but it also creates parallelism, which a batch dimension supplies for free.
Once the batch is wide the second job is already done and the reduction's barriers stop
paying for themselves -- measurably so: with the batch ignored, this kernel was up to 2.4x
slower than one work item per row at `nbatch >= 100` on eight of forty MIPLIB instances.

Narrowing is deliberately confined to the middle of the range, between [`MIN_BATCHED`](@ref)
and [`MAX_BATCHED`](@ref) work items per row. Below it sit matrices whose rows are already
barely worth splitting, where dropping further means falling back to one work item per row
and losing badly on the ones with a long row hiding behind a short average; above it sit
matrices of genuinely long rows, which keep wanting every lane they can get however wide
the batch is -- narrowing those cost 2.2x on the worst of them.
"""
function subgroup_size(A::GPUSparseMatrixCSR, nbatch::Integer)
    S = subgroup_size(A)
    S > MAX_BATCHED && return S
    shift = trailing_zeros(nextpow(2, max(nbatch, 1))) ÷ BATCH_SHIFT
    # `MIN_BATCHED` bounds how far the sub-group narrows, so it must never widen one that
    # already starts out below it
    return max(S >> shift, min(S, MIN_BATCHED))
end

"""
    spmv_csr!(c, A_rowptr, A_colval, A_nzval, b, α, β)

One work item per row. Used when the rows are too short for [`spmv_csr_vector!`](@ref)'s
sub-groups to pay for themselves.
"""
@kernel function spmv_csr!(
        c::DenseVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i = @index(Global, Linear)
    s = zero(T)
    @inbounds for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        s += A_nzval[k] * b[A_colval[k]]
    end
    @inbounds c[i] = α * s + β * c[i]
end

"""
    spmv_csr_vector!(c, A_rowptr, A_colval, A_nzval, b, α, β, Val(S))

`S` work items per row, reduced through local memory. Launched over `S * size(A, 1)` work
items rounded up to a whole number of workgroups, so it must check `row` against the number
of rows itself.

The reduction uses local memory rather than sub-group shuffles, which keeps it backend
agnostic at the cost of a few barriers. `@synchronize` splits the kernel into regions on
the CPU backend, and only `@uniform` values, `@localmem` arrays and top-level
`x = @index(...)` statements survive across them, which is why the indices below are
recomputed in each region instead of being carried over.
"""
@kernel function spmv_csr_vector!(
        c::DenseVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
        β::Number,
        ::Val{S}
    ) where {T, Ti, S}
    gid = @index(Global, Linear)
    lid = @index(Local, Linear)
    @uniform G = prod(@groupsize())
    @uniform nsteps = trailing_zeros(S)
    @uniform m = length(c)
    tmp = @localmem T (G,)

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    s = zero(T)
    if row <= m
        @inbounds for k in (A_rowptr[row] + Ti(lane)):Ti(S):(A_rowptr[row + Ti(1)] - Ti(1))
            s += A_nzval[k] * b[A_colval[k]]
        end
    end
    @inbounds tmp[lid] = s
    @synchronize

    for u in 1:nsteps
        lane = (gid - 1) % S
        span = S >> u
        @inbounds if lane < span
            tmp[lid] += tmp[lid + span]
        end
        @synchronize
    end

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    if lane == 0 && row <= m
        @inbounds c[row] = α * tmp[lid] + β * c[row]
    end
end

"""
    launch_spmv_csr!(c, A, b, α, β, backend, Val(S))

Launch [`spmv_csr_vector!`](@ref) with `S` work items per row.
"""
function launch_spmv_csr!(
        c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend, ::Val{S}
    ) where {S}
    kernel! = spmv_csr_vector!(backend, CSR_WORKGROUP)
    ndrange = cld(size(A, 1) * S, CSR_WORKGROUP) * CSR_WORKGROUP
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β, Val(S); ndrange)
    return nothing
end

"""
    launch_spmv_csr!(c, A, b, α, β, backend)

Launch the SpMV kernel best suited to `A`'s rows, per [`subgroup_size`](@ref).

The sub-group size has to reach the kernel as a `Val` so that the reduction unrolls and the
local memory can be sized, hence the chain of branches over the handful of sizes
[`subgroup_size`](@ref) can return.
"""
function launch_spmv_csr!(c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend)
    S = subgroup_size(A)
    if S >= 32
        launch_spmv_csr!(c, A, b, α, β, backend, Val(32))
    elseif S >= 16
        launch_spmv_csr!(c, A, b, α, β, backend, Val(16))
    elseif S >= 8
        launch_spmv_csr!(c, A, b, α, β, backend, Val(8))
    elseif S >= 4
        launch_spmv_csr!(c, A, b, α, β, backend, Val(4))
    elseif S >= 2
        launch_spmv_csr!(c, A, b, α, β, backend, Val(2))
    else
        kernel! = spmv_csr!(backend)
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    end
    return nothing
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixCSR{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        launch_spmv_csr!(c, A, b, One(), Zero(), backend)
    elseif α_is_one
        launch_spmv_csr!(c, A, b, One(), β, backend)
    elseif β_is_zero
        launch_spmv_csr!(c, A, b, α, Zero(), backend)
    else
        launch_spmv_csr!(c, A, b, α, β, backend)
    end
    return c
end

"""
    spmm_csr!(c, A_rowptr, A_colval, A_nzval, b, α, β)

Batched counterpart of [`spmv_csr!`](@ref): one work item per (row, batch column) pair.
"""
@kernel function spmm_csr!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, q = @index(Global, NTuple)
    s = zero(T)
    @inbounds for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        s += A_nzval[k] * b[A_colval[k], q]
    end
    @inbounds c[i, q] = α * s + β * c[i, q]
end

"""
    spmm_csr_vector!(c, A_rowptr, A_colval, A_nzval, b, α, β, Val(S))

Batched counterpart of [`spmv_csr_vector!`](@ref): `S` work items cooperate on one row of
one batch column. Launched over `(S * size(A, 1), size(c, 2))` with the cooperating lanes
contiguous in the first dimension, so that the workgroup is one-dimensional and its local
memory can be indexed by the linear local index.
"""
@kernel function spmm_csr_vector!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number,
        ::Val{S}
    ) where {T, Ti, S}
    gid, q = @index(Global, NTuple)
    lid = @index(Local, Linear)
    @uniform G = prod(@groupsize())
    @uniform nsteps = trailing_zeros(S)
    @uniform m = size(c, 1)
    tmp = @localmem T (G,)

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    s = zero(T)
    if row <= m
        @inbounds for k in (A_rowptr[row] + Ti(lane)):Ti(S):(A_rowptr[row + Ti(1)] - Ti(1))
            s += A_nzval[k] * b[A_colval[k], q]
        end
    end
    @inbounds tmp[lid] = s
    @synchronize

    for u in 1:nsteps
        lane = (gid - 1) % S
        span = S >> u
        @inbounds if lane < span
            tmp[lid] += tmp[lid + span]
        end
        @synchronize
    end

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    if lane == 0 && row <= m
        @inbounds c[row, q] = α * tmp[lid] + β * c[row, q]
    end
end

"""
    launch_spmm_csr!(c, A, b, α, β, backend, Val(S))

Launch [`spmm_csr_vector!`](@ref) with `S` work items per row.
"""
function launch_spmm_csr!(
        c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend, ::Val{S}
    ) where {S}
    kernel! = spmm_csr_vector!(backend, (CSR_WORKGROUP, 1))
    rows = cld(size(A, 1) * S, CSR_WORKGROUP) * CSR_WORKGROUP
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β, Val(S); ndrange = (rows, size(c, 2)))
    return nothing
end

"""
    launch_spmm_csr!(c, A, b, α, β, backend)

Launch the SpMM kernel best suited to `A`'s rows, per [`subgroup_size`](@ref).

The sub-group is sized from the batch as well as the matrix, per
[`subgroup_size(A, nbatch)`](@ref). Sub-groups of two are not worth their barriers here, so
anything below four work items per row -- which a wide batch will often produce -- falls
back to [`spmm_csr!`](@ref).
"""
function launch_spmm_csr!(c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend)
    S = subgroup_size(A, size(c, 2))
    if S >= 32
        launch_spmm_csr!(c, A, b, α, β, backend, Val(32))
    elseif S >= 16
        launch_spmm_csr!(c, A, b, α, β, backend, Val(16))
    elseif S >= 8
        launch_spmm_csr!(c, A, b, α, β, backend, Val(8))
    elseif S >= 4
        launch_spmm_csr!(c, A, b, α, β, backend, Val(4))
    else
        kernel! = spmm_csr!(backend)
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    end
    return nothing
end

function LinearAlgebra.mul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixCSR{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        launch_spmm_csr!(c, A, b, One(), Zero(), backend)
    elseif α_is_one
        launch_spmm_csr!(c, A, b, One(), β, backend)
    elseif β_is_zero
        launch_spmm_csr!(c, A, b, α, Zero(), backend)
    else
        launch_spmm_csr!(c, A, b, α, β, backend)
    end
    return c
end

## A CSR product that Reactant can raise to StableHLO

"""
    BLOCK_FANOUT

Number of entries that each block of [`spmv_csr_blocks!`](@ref) sums, i.e. the factor by which
each level of its pyramid is shorter than the level below.
"""
const BLOCK_FANOUT = 8

"""
    block_levels(nz, C)

Number of levels in a pyramid of block sums over `nz` entries with fan-out `C`, counting its
base: enough for [`peel_blocks`](@ref) to have consumed any range of these entries by the time
it leaves the top level.
"""
function block_levels(nz::Integer, C::Integer)
    L, width = 1, C
    while width <= nz
        L += 1
        width *= C
    end
    return L
end

"""
    CSRProducts(A_colval, A_nzval, b)

The products `A_nzval[k] * b[A_colval[k]]`, which index like a vector but are never stored:
they are the base of the pyramid of [`spmv_csr_blocks!`](@ref), of which a row reads at most
`2 * (BLOCK_FANOUT - 1)` entries.
"""
struct CSRProducts{Vi, V, Vb}
    A_colval::Vi
    A_nzval::V
    b::Vb
end

Base.@propagate_inbounds function Base.getindex(p::CSRProducts, k::Integer)
    return p.A_nzval[k] * p.b[p.A_colval[k]]
end

"""
    csr_block_products!(P, A_colval, A_nzval, b, Val(C))

Sum the products `A_nzval[k] * b[A_colval[k]]` over consecutive blocks of `C` nonzeros into `P`,
with one work item per block.
"""
@kernel function csr_block_products!(
        P::DenseVector{T},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        ::Val{C}
    ) where {T, Ti, C}
    q = @index(Global, Linear)
    nz = length(A_nzval)
    s = zero(T)
    for u in 1:C
        k = (q - 1) * C + u
        if k <= nz
            @inbounds s += A_nzval[k] * b[A_colval[k]]
        end
    end
    @inbounds P[q] = s
end

"""
    block_sums!(Q, P, Val(C))

Sum `P` over consecutive blocks of `C` entries into `Q`, with one work item per block.
"""
@kernel function block_sums!(Q::DenseVector{T}, P::DenseVector{T}, ::Val{C}) where {T, C}
    q = @index(Global, Linear)
    n = length(P)
    s = zero(T)
    for u in 1:C
        x = (q - 1) * C + u
        @inbounds v = x <= n ? P[x] : zero(T)
        s += v
    end
    @inbounds Q[q] = s
end

"""
    peel_blocks(P, lo, hi, s, Val(C))

Add to `s` the entries of `P` in `lo:hi` that do not fill a whole block of `C`: at most `C - 1`
from `lo` up to the first block boundary, and as many from `hi` down to the last one. Return
the whole blocks that remain, as a range of indices one level up the pyramid, and the new sum.

Every remainder here is of a non-negative value: once a kernel is raised to StableHLO, Reactant
computes `mod` and `fld` of a negative value as if they truncated (see
https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl/issues/167).
"""
@inline function peel_blocks(P, lo::Int, hi::Int, s, ::Val{C}) where {C}
    r = (lo - 1) % C
    nleft = max(min(hi - lo + 1, ifelse(r == 0, 0, C - r)), 0)
    for u in 0:(C - 2)
        @inbounds v = u < nleft ? P[lo + u] : zero(s)
        s += v
    end
    lo += nleft
    nright = max(min(hi - lo + 1, hi % C), 0)
    for u in 0:(C - 2)
        @inbounds v = u < nright ? P[hi - u] : zero(s)
        s += v
    end
    hi -= nright
    return (lo - 1) ÷ C + 1, hi ÷ C, s
end

"""
    sum_blocks(levels, lo, hi, s, Val(C))

Add to `s` the sum of `first(levels)[lo:hi]`, read from the coarsest blocks of the pyramid
`levels` that fit in this range.
"""
@inline sum_blocks(::Tuple{}, lo::Int, hi::Int, s, ::Val{C}) where {C} = s
@inline function sum_blocks(levels::Tuple, lo::Int, hi::Int, s, ::Val{C}) where {C}
    lo, hi, s = peel_blocks(first(levels), lo, hi, s, Val(C))
    return sum_blocks(Base.tail(levels), lo, hi, s, Val(C))
end

"""
    csr_block_rows!(c, A_rowptr, A_colval, A_nzval, b, levels, α, β, Val(C))

Set `c[i] = α * s + β * c[i]`, where `s` sums the products of row `i` over the pyramid whose
base is [`CSRProducts`](@ref) and whose upper levels are `levels`: one work item per row.
"""
@kernel function csr_block_rows!(
        c::DenseVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        levels::Tuple,
        α::Number,
        β::Number,
        ::Val{C}
    ) where {T, Ti, C}
    i = @index(Global, Linear)
    @inbounds lo, hi = Int(A_rowptr[i]), Int(A_rowptr[i + Ti(1)]) - 1
    base = CSRProducts(A_colval, A_nzval, b)
    s = sum_blocks((base, levels...), lo, hi, zero(T), Val(C))
    @inbounds c[i] = α * s + β * c[i]
end

"""
    spmv_csr_blocks!(c, A::GPUSparseMatrixCSR, b, α, β)

Compute `c = α * A * b + β * c`, like `mul!`, with kernels whose every loop has a trip count
fixed at launch. This is what lets Reactant raise them to StableHLO (`@compile raise = true`),
which XLA then optimizes along with the rest of the program, and which backends that cannot
run a custom kernel require.

The loops of `mul!`'s kernels run over the nonzeros of a row, so their trip counts are read
from `A.rowptr`. Reactant either refuses to raise them (the local memory of
[`spmv_csr_vector!`](@ref)), or raises them into a loop over the longest row for every row at
once, which makes a product with a long row orders of magnitude slower.

The products are summed over a pyramid of blocks instead. Its base holds the products
`A.nzval[k] * b[A.colval[k]]` in storage order, and each level above it sums consecutive
blocks of [`BLOCK_FANOUT`](@ref) entries of the level below, across row boundaries. A row is
a range of the base, and its sum is read from the coarsest blocks that fit inside that range:
[`peel_blocks`](@ref) takes the entries at both ends that do not fill a whole block, and moves
the rest of the range up one level, until nothing is left. A block that fits inside the range
only holds products of that row, so nothing is ever subtracted and the other rows cannot
affect the result.

A row reads at most `2 * (BLOCK_FANOUT - 1)` entries per level, whatever its length, and the
number of levels grows with the logarithm of `nnz(A)`. The base is never stored: a row
recomputes the few products it reads (see [`CSRProducts`](@ref)), so the pyramid costs
little more than one pass over the nonzeros.

Every row goes through every level, so this is slowest on matrices with many short rows.
"""
function spmv_csr_blocks!(
        c::AbstractVector, A::GPUSparseMatrixCSR, b::AbstractVector, α::Number, β::Number
    )
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    m, nz = size(A, 1), nnz(A)
    C = BLOCK_FANOUT
    m == 0 && return c
    levels = typeof(A.nzval)[]
    for _ in 2:block_levels(nz, C)
        if isempty(levels)
            P = similar(A.nzval, cld(nz, C))
            csr_block_products!(backend)(P, A.colval, A.nzval, b, Val(C); ndrange = length(P))
        else
            Q = last(levels)
            P = similar(Q, cld(length(Q), C))
            block_sums!(backend)(P, Q, Val(C); ndrange = length(P))
        end
        push!(levels, P)
    end
    kernel! = csr_block_rows!(backend)
    args = (c, A.rowptr, A.colval, A.nzval, b, Tuple(levels))
    if isone(α) && iszero(β)
        kernel!(args..., One(), Zero(), Val(C); ndrange = m)
    elseif isone(α)
        kernel!(args..., One(), β, Val(C); ndrange = m)
    elseif iszero(β)
        kernel!(args..., α, Zero(), Val(C); ndrange = m)
    else
        kernel!(args..., α, β, Val(C); ndrange = m)
    end
    return c
end
