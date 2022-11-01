function contraction_sequence(tn::Vector{ITensor}; alg="optimal", kwargs...)
  return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
  seq_linear_index = contraction_sequence(Vector{ITensor}(tn); kwargs...)
  # TODO: use Functors.fmap
  return deepmap(n -> vertices(tn)[n], seq_linear_index)
end

function contraction_sequence(::Algorithm"optimal", tn::Vector{ITensor})
  return optimal_contraction_sequence(tn)
end

function contraction_sequence_requires_error(module_name, algorithm)
  return "Module `$(module_name)` not found, please type `using $(module_name)` before using the \"$(algorithm)\" contraction sequence backend!"
end

function contraction_sequence(::Algorithm"greedy", tn::Vector{ITensor}; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "greedy"))
  end
  return contraction_sequence(OMEinsumContractionOrders.GreedyMethod(; kwargs...), tn)
end

function contraction_sequence(::Algorithm"tree_sa", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "tree_sa"))
  end
  return contraction_sequence(OMEinsumContractionOrders.TreeSA(; kwargs...), tn)
end

function contraction_sequence(::Algorithm"sa_bipartite", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "sa_bipartite"))
  end
  return contraction_sequence(OMEinsumContractionOrders.SABipartite(; kwargs...), tn)
end

function contraction_sequence(::Algorithm"kahypar_bipartite", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "kahypar_bipartite"))
  end
  return contraction_sequence(OMEinsumContractionOrders.KaHyParBipartite(; kwargs...), tn)
end
