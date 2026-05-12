using ITensors.LazyApply: Applied, Prod, Scaled, Sum
using ITensors.Ops: Op, Ops
using ITensors: ITensor, filterinds, hascommoninds, op, uniqueinds

# Drop a spurious zero imaginary part on a complex scalar so that downstream
# scalar promotions stay real. Used when materializing `Scaled` `OpSum` terms
# and when applying scalar-multiplied gates in `apply.jl`.
maybe_real(x::Real) = x
maybe_real(x::Complex) = iszero(imag(x)) ? real(x) : x

# Tensor sum: `A ⊞ B = A ⊗ Iᴮ + Iᴬ ⊗ B`
# https://github.com/JuliaLang/julia/issues/13333#issuecomment-143825995
# "PRESERVATION OF TENSOR SUM AND TENSOR PRODUCT"
# C. S. KUBRUSLY and N. LEVAN
# https://www.emis.de/journals/AMUC/_vol-80/_no_1/_kubrusly/kubrusly.pdf
function tensor_sum(A::ITensor, B::ITensor)
    extend_A = filterinds(uniqueinds(B, A); plev = 0)
    extend_B = filterinds(uniqueinds(A, B); plev = 0)
    for i in extend_A
        A *= op("I", i)
    end
    for i in extend_B
        B *= op("I", i)
    end
    return A + B
end

function ITensors.ITensor(o::Op, s::IndsNetwork)
    s⃗ = [only(s[nᵢ]) for nᵢ in Ops.sites(o)]
    return op(Ops.which_op(o), s⃗; Ops.params(o)...)
end

function ITensors.ITensor(∏o::Prod, s::IndsNetwork)
    T = ITensor(1.0)
    for oᵢ in Ops.terms(∏o)
        Tᵢ = ITensor(oᵢ, s)
        # For now, only support operators on distinct
        # sites.
        @assert !hascommoninds(T, Tᵢ)
        T *= Tᵢ
    end
    return T
end

function ITensors.ITensor(∑o::Sum, s::IndsNetwork)
    T = ITensor(0)
    for oᵢ in Ops.terms(∑o)
        Tᵢ = ITensor(oᵢ, s)
        T = tensor_sum(T, Tᵢ)
    end
    return T
end

function ITensors.ITensor(o::Scaled, s::IndsNetwork)
    return maybe_real(Ops.coefficient(o)) * ITensor(Ops.argument(o), s)
end

function ITensors.ITensor(o::Ops.Exp, s::IndsNetwork)
    return exp(ITensor(Ops.argument(o), s))
end

function Base.Vector{ITensor}(o::Union{Sum, Prod}, s::IndsNetwork)
    T⃗ = ITensor[]
    for oᵢ in Ops.terms(o)
        Tᵢ = ITensor(oᵢ, s)
        T⃗ = [T⃗; Tᵢ]
    end
    return T⃗
end
