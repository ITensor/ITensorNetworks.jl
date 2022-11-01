function tebd(ℋ::Sum, ψ::AbstractITensorNetwork; β, Δβ, maxdim, cutoff, print_frequency=10, ortho=false)
  𝒰 = exp(-Δβ * ℋ; alg=Trotter{2}())
  # Imaginary time evolution terms
  s = siteinds(ψ)
  u⃗ = Vector{ITensor}(𝒰, s)
  nsteps = Int(β ÷ Δβ)
  for step in 1:nsteps
    if step % print_frequency == 0
      @show step, (step - 1) * Δβ, β
    end
    ψ = insert_links(ψ)
    ψ = apply(u⃗, ψ; cutoff, maxdim, normalize=true, ortho)
    if ortho
      for v in vertices(ψ)
        ψ = orthogonalize(ψ, v)
      end
    end
  end
  return ψ
end

# using ITensors
# using ITensorNetworks
# using Dictionaries
# using NamedGraphs
# using SplitApplyCombine
# using ITensors.ContractionSequenceOptimization
# 
# using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!

# to_tuple(x) = (x,)
# to_tuple(x::Tuple) = x

# function group_terms(ℋ::Sum, g)
#   grouped_terms = group(ITensors.terms(ℋ)) do t
#     findfirst(edges(g)) do e
#       to_tuple.(ITensors.sites(t)) ⊆ [src(e), dst(e)]
#     end
#   end
#   return Sum(collect(sum.(grouped_terms)))
# end

# function cartesian_to_linear(dims::Tuple)
#   return Dictionary(vec(Tuple.(CartesianIndices(dims))), 1:prod(dims))
# end

# NamedGraphs.NamedDimGraph(vertices::Vector) = NamedDimGraph(tuple.(vertices))
# NamedGraphs.NamedDimGraph(vertices::Vector{<:Tuple}) = NamedDimGraph(Graph(length(vertices)); vertices)
# 
# function rename_vertices(e::AbstractEdge, name_map::Dictionary)
#   return typeof(e)(name_map[src(e)], name_map[dst(e)])
# end
# 
# function rename_vertices(g::NamedDimGraph, name_map::Dictionary)
#   original_vertices = vertices(g)
#   new_vertices = getindices(name_map, original_vertices)
#   new_g = NamedDimGraph(new_vertices)
#   for e in edges(g)
#     add_edge!(new_g, rename_vertices(e, name_map))
#   end
#   return new_g
# end
# 
# function rename_vertices(g::NamedDimGraph, name_map::Function)
#   original_vertices = vertices(g)
#   return rename_vertices(g, Dictionary(original_vertices, name_map.(original_vertices)))
# end

# # Convert to real if possible
# maybe_real(x::Real) = x
# maybe_real(x::Complex) = iszero(imag(x)) ? real(x) : x

# function ITensors.ITensor(o::Op, s::IndsNetwork)
#   s⃗ = [only(s[nᵢ]) for nᵢ in Ops.sites(o)]
#   return op(Ops.which_op(o), s⃗; Ops.params(o)...)
# end
# 
# function ITensors.ITensor(∏o::Prod, s::IndsNetwork)
#   T = ITensor(1.0)
#   for oᵢ in Ops.terms(∏o)
#     Tᵢ = ITensor(oᵢ, s)
#     # For now, only support operators on distinct
#     # sites.
#     @assert !hascommoninds(T, Tᵢ)
#     T *= Tᵢ
#   end
#   return T
# end

# # Tensor sum: `A ⊞ B = A ⊗ Iᴮ + Iᴬ ⊗ B`
# # https://github.com/JuliaLang/julia/issues/13333#issuecomment-143825995
# # "PRESERVATION OF TENSOR SUM AND TENSOR PRODUCT"
# # C. S. KUBRUSLY and N. LEVAN
# # https://www.emis.de/journals/AMUC/_vol-80/_no_1/_kubrusly/kubrusly.pdf
# function tensor_sum(A::ITensor, B::ITensor)
#   extend_A = filterinds(uniqueinds(B, A); plev=0)
#   extend_B = filterinds(uniqueinds(A, B); plev=0)
#   for i in extend_A
#     A *= op("I", i)
#   end
#   for i in extend_B
#     B *= op("I", i)
#   end
#   return A + B
# end

# function ITensors.ITensor(∑o::Sum, s::IndsNetwork)
#   T = ITensor(0)
#   for oᵢ in Ops.terms(∑o)
#     Tᵢ = ITensor(oᵢ, s)
#     T = tensor_sum(T, Tᵢ)
#   end
#   return T
# end
# 
# function ITensors.ITensor(o::Scaled, s::IndsNetwork)
#   return maybe_real(Ops.coefficient(o)) * ITensor(Ops.argument(o), s)
# end
# 
# function ITensors.ITensor(o::Ops.Exp, s::IndsNetwork)
#   return exp(ITensor(Ops.argument(o), s))
# end
# 
# function Base.Vector{ITensor}(o::Union{Sum,Prod}, s::IndsNetwork)
#   T⃗ = ITensor[]
#   for oᵢ in Ops.terms(o)
#     Tᵢ = ITensor(oᵢ, s)
#     T⃗ = [T⃗; Tᵢ]
#   end
#   return T⃗
# end

# using ITensorNetworks: ⊔

# function neighbor_vertices(ψ::ITensorNetwork, T::ITensor)
#   ψT = ψ ⊔ ITensorNetwork([T])
#   v⃗ = neighbors(ψT, (2, 1))
#   return Base.tail.(v⃗)
# end

# function ITensors.orthogonalize(ψ::ITensorNetwork, source_vertex::Tuple)
#   spanning_tree_edges = post_order_dfs_edges(bfs_tree(ψ, source_vertex), source_vertex)
#   for e in spanning_tree_edges
#     ψ = orthogonalize(ψ, e)
#   end
#   return ψ
# end

# function ITensors.apply(o::ITensor, ψ::ITensorNetwork; cutoff, maxdim, normalize=false, ortho=false)
#   ψ = copy(ψ)
#   v⃗ = neighbor_vertices(ψ, o)
#   if length(v⃗) == 1
#     if ortho
#       ψ = orthogonalize(ψ, v⃗[1])
#     end
#     oψᵥ = apply(o, ψ[v⃗[1]])
#     if normalize
#       oψᵥ ./= norm(oψᵥ)
#     end
#     ψ[v⃗[1]] = oψᵥ
#   elseif length(v⃗) == 2
#     e = v⃗[1] => v⃗[2]
#     if !has_edge(ψ, e)
#       error("Vertices where the gates are being applied must be neighbors for now.")
#     end
#     if ortho
#       ψ = orthogonalize(ψ, v⃗[1])
#     end
#     oψᵥ = apply(o, ψ[v⃗[1]] * ψ[v⃗[2]])
#     ψᵥ₁, ψᵥ₂ = factorize(oψᵥ, inds(ψ[v⃗[1]]); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e))
#     if normalize
#       ψᵥ₁ ./= norm(ψᵥ₁)
#       ψᵥ₂ ./= norm(ψᵥ₂)
#     end
#     ψ[v⃗[1]] = ψᵥ₁
#     ψ[v⃗[2]] = ψᵥ₂
#   elseif length(v⃗) < 1
#     error("Gate being applied does not share indices with tensor network.")
#   elseif length(v⃗) > 2
#     error("Gates with more than 2 sites is not supported yet.")
#   end
#   return ψ
# end

# function ITensors.apply(o⃗::Vector{ITensor}, ψ::ITensorNetwork; cutoff, maxdim, normalize=false, ortho=false)
#   o⃗ψ = ψ
#   for oᵢ in o⃗
#     o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
#   end
#   return o⃗ψ
# end
# 
# function ITensors.apply(o⃗::Scaled, ψ::ITensorNetwork; cutoff, maxdim, normalize=false, ortho=false)
#   return maybe_real(Ops.coefficient(o⃗)) * apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho)
# end

# function Base.:*(c::Number, ψ::ITensorNetwork)
#   v₁ = first(vertices(ψ))
#   cψ = copy(ψ)
#   cψ[v₁] *= c
#   return cψ
# end

# function ITensors.apply(o⃗::Prod, ψ::ITensorNetwork; cutoff, maxdim, normalize=false, ortho=false)
#   o⃗ψ = ψ
#   for oᵢ in o⃗
#     o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
#   end
#   return o⃗ψ
# end
# 
# function ITensors.apply(o::Op, ψ::ITensorNetwork; cutoff, maxdim, normalize=false, ortho=false)
#   return apply(ITensor(o, siteinds(ψ)), ψ; cutoff, maxdim, normalize, ortho)
# end

# function flattened_inner_network(ϕ::ITensorNetwork, ψ::ITensorNetwork)
#   tn = inner(prime(ϕ; sites=[]), ψ)
#   for v in vertices(ψ)
#     tn = ITensors.contract(tn, (2, v...) => (1, v...))
#   end
#   return tn
# end
# 
# function contract_inner(ϕ::ITensorNetwork, ψ::ITensorNetwork; sequence=nothing)
#   tn = inner(prime(ϕ; sites=[]), ψ)
#   # TODO: convert to an IndsNetwork and compute the contraction sequence
#   for v in vertices(ψ)
#     tn = ITensors.contract(tn, (2, v...) => (1, v...))
#   end
#   if isnothing(sequence)
#     sequence = optimal_contraction_sequence(tn)
#   end
#   return ITensors.contract(tn; sequence)[]
# end

# norm2(ψ::ITensorNetwork; sequence) = contract_inner(ψ, ψ; sequence)

# function ITensors.expect(op::String, ψ::ITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=false, sequence=nothing)
#   res = Dictionary(vertices(ψ), Vector{Float64}(undef, nv(ψ)))
#   if isnothing(sequence)
#     sequence = optimal_contraction_sequence(flattened_inner_network(ψ, ψ))
#   end
#   normψ² = norm2(ψ; sequence)
#   for v in vertices(ψ)
#     O = ITensor(Op(op, v), s)
#     Oψ = apply(O, ψ; cutoff, maxdim, ortho)
#     res[v] = contract_inner(ψ, Oψ; sequence) / normψ²
#   end
#   return res
# end
# 
# function ITensors.expect(ℋ::OpSum, ψ::ITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=false, sequence=nothing)
#   s = siteinds(ψ)
#   # h⃗ = Vector{ITensor}(ℋ, s)
#   if isnothing(sequence)
#     sequence = optimal_contraction_sequence(flattened_inner_network(ψ, ψ))
#   end
#   h⃗ψ = [apply(hᵢ, ψ; cutoff, maxdim, ortho) for hᵢ in ITensors.terms(ℋ)]
#   ψhᵢψ = [contract_inner(ψ, hᵢψ; sequence) for hᵢψ in h⃗ψ]
#   ψh⃗ψ = sum(ψhᵢψ)
#   ψψ = norm2(ψ; sequence)
#   return ψh⃗ψ / ψψ
# end

# function ITensors.expect(opsum_sum::Sum{<:OpSum}, ψ::ITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=true, sequence=nothing)
#   return expect(sum(Ops.terms(opsum_sum)), ψ; cutoff, maxdim, ortho, sequence)
# end

# function randomITensorNetwork(s; link_space)
#   ψ = ITensorNetwork(s; link_space)
#   for v in vertices(ψ)
#     ψᵥ = copy(ψ[v])
#     randn!(ψᵥ)
#     ψᵥ ./= norm(ψᵥ)
#     ψ[v] = ψᵥ
#   end
#   return ψ
# end

# function ITensors.MPO(opsum::OpSum, s::IndsNetwork)
#   s_linear = [only(s[v]) for v in 1:nv(s)]
#   return MPO(opsum, s_linear)
# end
# 
# function ITensors.MPO(opsum_sum::Sum{<:OpSum}, s::IndsNetwork)
#   return MPO(sum(Ops.terms(opsum_sum)), s)
# end

# function ITensors.randomMPS(s::IndsNetwork, args...; kwargs...)
#   s_linear = [only(s[v]) for v in 1:nv(s)]
#   return randomMPS(s_linear, args...; kwargs...)
# end
# 
# function ITensors.MPS(s::IndsNetwork, args...; kwargs...)
#   s_linear = [only(s[v]) for v in 1:nv(s)]
#   return MPS(s_linear, args...; kwargs...)
# end

# maybe_only(x) = x
# maybe_only(x::Tuple{T}) where {T} = only(x)

# function ising(g::AbstractGraph; h)
#   ℋ = OpSum()
#   for e in edges(g)
#     ℋ -= "Z", maybe_only(src(e)), "Z", maybe_only(dst(e))
#   end
#   for v in vertices(g)
#     ℋ += h, "X", maybe_only(v)
#   end
#   return ℋ
# end

