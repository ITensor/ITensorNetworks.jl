#
# ITensors.jl extensions
#

# Generalize siteind to n-dimensional lattice
function ITensors.siteind(st::SiteType, N1::Integer, N2::Integer, Ns::Integer...; kwargs...)
  s = siteind(st; kwargs...)
  if !isnothing(s)
    ts = "n1=$N1,n2=$N2"
    for i in eachindex(Ns)
      ts *= ",n$(i + 2)=$(Ns[i])"
    end
    return addtags(s, ts)
  end
  return isnothing(s) && error(space_error_message(st))
end

# Generalize siteinds to n-dimensional lattice
function ITensors.siteinds(
  str::AbstractString, N1::Integer, N2::Integer, Ns::Integer...; kwargs...
)
  st = SiteType(str)
  return [siteind(st, ns...) for ns in Base.product(1:N1, 1:N2, UnitRange.(1, Ns)...)]
end

# Get the promoted type of the Index objects in a collection
# of Index (Tuple, Vector, ITensor, etc.)
indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{<:Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

#
# MPS functionality extensions
#

Base.keytype(m::MPS) = keytype(data(m))

# A version of indexing which returns an empty order-0 ITensor
# when out of bounds
get_itensor(x::MPS, n::Int) = n in 1:length(x) ? x[n] : ITensor()

# Reverse the site ordering of an MPS.
# XXX: also reverse the orthogonality limits.
Base.reverse(x::MPS) = MPS(reverse(x.data))
