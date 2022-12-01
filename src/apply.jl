function ITensors.apply(
  o::ITensor, ψ::AbstractITensorNetwork; cutoff=nothing, maxdim=nothing, normalize=false, ortho=false
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, o)
  if length(v⃗) == 1
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    oψᵥ = apply(o, ψ[v⃗[1]])
    if normalize
      oψᵥ ./= norm(oψᵥ)
    end
    ψ[v⃗[1]] = oψᵥ
  elseif length(v⃗) == 2
    e = v⃗[1] => v⃗[2]
    if !has_edge(ψ, e)
      error("Vertices where the gates are being applied must be neighbors for now.")
    end
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    oψᵥ = apply(o, ψ[v⃗[1]] * ψ[v⃗[2]])
    ψᵥ₁, ψᵥ₂ = factorize(
      oψᵥ, inds(ψ[v⃗[1]]); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e)
    )
    if normalize
      ψᵥ₁ ./= norm(ψᵥ₁)
      ψᵥ₂ ./= norm(ψᵥ₂)
    end
    ψ[v⃗[1]] = ψᵥ₁
    ψ[v⃗[2]] = ψᵥ₂
  elseif length(v⃗) < 1
    error("Gate being applied does not share indices with tensor network.")
  elseif length(v⃗) > 2
    error("Gates with more than 2 sites is not supported yet.")
  end
  return ψ
end

function ITensors.apply(
  o⃗::Vector{ITensor},
  ψ::AbstractITensorNetwork;
  cutoff,
  maxdim,
  normalize=false,
  ortho=false,
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o⃗::Scaled, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return maybe_real(Ops.coefficient(o⃗)) *
         apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho)
end

function ITensors.apply(
  o⃗::Prod, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o::Op, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return apply(ITensor(o, siteinds(ψ)), ψ; cutoff, maxdim, normalize, ortho)
end
