# convert ITensors.OpSum to TreeTensorNetwork

# 
# Utility methods
# 

# linear ordering of vertices in tree graph relative to chosen root, chosen outward from root
function find_index_in_tree(site, g::AbstractGraph, root_vertex)
  ordering = reverse(post_order_dfs_vertices(g, root_vertex))
  return findfirst(x -> x == site, ordering)
end
function find_index_in_tree(o::Op, g::AbstractGraph, root_vertex)
  return find_index_in_tree(ITensors.site(o), g, root_vertex)
end

# determine 'support' of product operator on tree graph
function span(t::Scaled{C,Prod{Op}}, g::AbstractGraph) where {C}
  spn = eltype(g)[]
  nterms = length(t)
  for i in 1:nterms, j in i:nterms
    path = vertex_path(g, ITensors.site(t[i]), ITensors.site(t[j]))
    spn = union(spn, path)
  end
  return spn
end

# determine whether an operator string crosses a given graph vertex
function crosses_vertex(t::Scaled{C,Prod{Op}}, g::AbstractGraph, v) where {C}
  return v ∈ span(t, g)
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo.jl
# 

"""
    svdTTN(os::OpSum{C}, sites::IndsNetwork{V<:Index}, root_vertex::V, kwargs...) where {C,V}

Construct a dense TreeTensorNetwork from a symbolic OpSum representation of a
Hamiltonian, compressing shared interaction channels.
"""
function svdTTN(
  os::OpSum{C}, sites::IndsNetwork{VT,<:Index}, root_vertex::VT; kwargs...
)::TTN where {C,VT}
  mindim::Int = get(kwargs, :mindim, 1)
  maxdim::Int = get(kwargs, :maxdim, 10000)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-15)

  ValType = ITensors.determineValType(ITensors.terms(os))

  # traverse tree outwards from root vertex
  vs = reverse(post_order_dfs_vertices(sites, root_vertex))                                 # store vertices in fixed ordering relative to root
  es = reverse(reverse.(post_order_dfs_edges(sites, root_vertex)))                          # store edges in fixed ordering relative to root
  # some things to keep track of
  ranks = Dict(v => degree(sites, v) for v in vs)                                           # rank of every TTN tensor in network
  Vs = Dict(e => Matrix{ValType}(undef, 1, 1) for e in es)                                  # link isometries for SVD compression of TTN
  inmaps = Dict(e => Dict{Vector{Op},Int}() for e in es)                                    # map from term in Hamiltonian to incoming channel index for every edge
  outmaps = Dict(e => Dict{Vector{Op},Int}() for e in es)                                   # map from term in Hamiltonian to outgoing channel index for every edge
  inbond_coefs = Dict(e => ITensors.MatElem{ValType}[] for e in es)                         # bond coefficients for incoming edge channels
  site_coef_done = Prod{Op}[]                                                               # list of terms for which the coefficient has been added to a site factor

  # temporary symbolic representation of TTN Hamiltonian
  tempTTN = Dict(v => ArrElem{Scaled{C,Prod{Op}},ranks[v]}[] for v in vs)

  # build compressed finite state machine representation
  for v in vs
    # for every vertex, find all edges that contain this vertex
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    # use the corresponding ordering as index order for tensor elements at this site
    dim_in = findfirst(e -> dst(e) == v, edges)
    edge_in = (isnothing(dim_in) ? [] : edges[dim_in])
    dims_out = findall(e -> src(e) == v, edges)
    edges_out = edges[dims_out]

    # sanity check, leaves only have single incoming or outgoing edge
    @assert !isempty(dims_out) || !isnothing(dim_in)
    (isempty(dims_out) || isnothing(dim_in)) && @assert is_leaf(sites, v)

    for term in os
      # loop over OpSum and pick out terms that act on current vertex
      crosses_vertex(term, sites, v) || continue

      # filter out factors that come in from the direction of the incoming edge
      incoming = filter(
        t -> edge_in ∈ edge_path(sites, ITensors.site(t), v), ITensors.terms(term)
      )
      # also store all non-incoming factors in standard order, used for channel merging
      not_incoming = filter(
        t -> edge_in ∉ edge_path(sites, ITensors.site(t), v), ITensors.terms(term)
      )
      # filter out factor that acts on current vertex
      onsite = filter(t -> (ITensors.site(t) == v), ITensors.terms(term))
      # for every outgoing edge, filter out factors that go out along that edge
      outgoing = Dict(
        e => filter(t -> e ∈ edge_path(sites, v, ITensors.site(t)), ITensors.terms(term))
        for e in edges_out
      )

      # translate into tensor entry
      T_inds = MVector{ranks[v]}(fill(-1, ranks[v]))
      bond_row = -1
      bond_col = -1
      if !isempty(incoming)
        bond_row = ITensors.posInLink!(inmaps[edge_in], incoming)
        bond_col = ITensors.posInLink!(outmaps[edge_in], not_incoming) # get incoming channel
        bond_coef = convert(ValType, ITensors.coefficient(term))
        push!(inbond_coefs[edge_in], ITensors.MatElem(bond_row, bond_col, bond_coef))
        T_inds[dim_in] = bond_col
      end
      for dout in dims_out
        T_inds[dout] = ITensors.posInLink!(outmaps[edges[dout]], outgoing[edges[dout]]) # add outgoing channel
      end
      # if term starts at this site, add its coefficient as a site factor
      site_coef = one(C)
      if (isnothing(dim_in) || T_inds[dim_in] == -1) &&
        ITensors.argument(term) ∉ site_coef_done
        site_coef = ITensors.coefficient(term)
        push!(site_coef_done, ITensors.argument(term))
      end
      # add onsite identity for interactions passing through vertex
      if isempty(onsite)
        if !ITensors.using_auto_fermion() && isfermionic(incoming, sites)
          error("No verified fermion support for automatic TTN constructor!")
        else
          push!(onsite, Op("Id", v))
        end
      end
      # save indices and value of symbolic tensor entry
      el = ArrElem(T_inds, site_coef * Prod(onsite))
      push!(tempTTN[v], el)
    end
    ITensors.remove_dups!(tempTTN[v])
    # manual truncation: isometry on incoming edge
    if !isnothing(dim_in) && !isempty(inbond_coefs[edges[dim_in]])
      M = ITensors.toMatrix(inbond_coefs[edges[dim_in]])
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[edges[dim_in]] = Matrix{ValType}(V[1:nc, 1:tdim])
    end
  end

  # compress this tempTTN representation into dense form

  link_space = dictionary([
    e => Index((isempty(outmaps[e]) ? 0 : size(Vs[e], 2)) + 2, edge_tag(e)) for e in es
  ])

  H = TTN(sites)

  for v in vs

    # redo the whole thing like before
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    dim_in = findfirst(e -> dst(e) == v, edges)
    dims_out = findall(e -> src(e) == v, edges)

    # slice isometries at this vertex
    Vv = [Vs[e] for e in edges]

    linkinds = [link_space[e] for e in edges]
    linkdims = dim.(linkinds)

    H[v] = ITensor()

    for el in tempTTN[v]
      T_inds = el.idxs
      t = el.val
      (abs(coefficient(t)) > eps()) || continue
      T = zeros(ValType, linkdims...)
      ct = convert(ValType, coefficient(t))
      terminal_dims = findall(d -> T_inds[d] == -1, 1:ranks[v])   # directions in which term starts or ends
      normal_dims = findall(d -> T_inds[d] ≠ -1, 1:ranks[v])      # normal dimensions, do truncation thingies
      T_inds[terminal_dims] .= 1                                  # start in channel 1
      for dout in filter(d -> d ∈ terminal_dims, dims_out)
        T_inds[dout] = linkdims[dout]                             # end in channel linkdims[d] for each dimension d
      end
      if isempty(normal_dims)
        T[T_inds...] += ct                                        # on-site term
      else
        # handle channel compression isometries
        dim_ranges = Tuple(size(Vv[d], 2) for d in normal_dims)
        for c in CartesianIndices(dim_ranges)
          z = ct
          temp_inds = copy(T_inds)
          for (i, d) in enumerate(normal_dims)
            V_factor = Vv[d][T_inds[d], c[i]]
            z *= (d == dim_in ? conj(V_factor) : V_factor) # conjugate incoming isemetry factor
            temp_inds[d] = 1 + c[i]
          end
          T[temp_inds...] += z
        end
      end
      T = itensor(T, linkinds)
      H[v] += T * computeSiteProd(sites, ITensors.argument(t))
    end

    # add starting and ending identity operators
    idT = zeros(ValType, linkdims...)
    if isnothing(dim_in)
      idT[ones(Int, ranks[v])...] = 1.0 # only one real starting identity
    end
    # ending identities are a little more involved
    if !isnothing(dim_in)
      idT[linkdims...] = 1.0 # place identity if all channels end
      # place identity from start of incoming channel to start of each single outgoing channel, and end all other channels
      idT_end_inds = [linkdims...]
      idT_end_inds[dim_in] = 1.0
      for dout in dims_out
        idT_end_inds[dout] = 1.0
        idT[idT_end_inds...] = 1.0
        idT_end_inds[dout] = linkdims[dout] # reset
      end
    end
    T = itensor(idT, linkinds)
    H[v] += T * ITensorNetworks.computeSiteProd(sites, Prod([Op("Id", v)]))
  end

  return H
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo_generic.jl
# 

# TODO: fix quantum number and fermion support, definitely broken

# needed an extra `only` compared to ITensors version since IndsNetwork has Vector{<:Index}
# as vertex data
function isfermionic(t::Vector{Op}, sites::IndsNetwork{V,<:Index}) where {V}
  p = +1
  for op in t
    if has_fermion_string(ITensors.name(op), only(sites[ITensors.site(op)]))
      p *= -1
    end
  end
  return (p == -1)
end

# only(site(ops[1])) in ITensors breaks for Tuple site labels, had to drop the only
function computeSiteProd(sites::IndsNetwork{V,<:Index}, ops::Prod{Op})::ITensor where {V}
  v = ITensors.site(ops[1])
  T = op(sites[v], ITensors.which_op(ops[1]); ITensors.params(ops[1])...)
  for j in 2:length(ops)
    (ITensors.site(ops[j]) != v) && error("Mismatch of vertex labels in computeSiteProd")
    opj = op(sites[v], ITensors.which_op(ops[j]); ITensors.params(ops[j])...)
    T = product(T, opj)
  end
  return T
end

# changed `isless_site` to use tree vertex ordering relative to root
function sorteachterm(os::OpSum, sites::IndsNetwork{V,<:Index}, root_vertex::V) where {V}
  os = copy(os)
  findpos(op::Op) = find_index_in_tree(op, sites, root_vertex)
  isless_site(o1::Op, o2::Op) = findpos(o1) < findpos(o2)
  N = nv(sites)
  for n in eachindex(os)
    t = os[n]
    Nt = length(t)

    if !all(map(v -> has_vertex(sites, v), ITensors.sites(t)))
      error(
        "The OpSum contains a term $t that does not have support on the underlying graph."
      )
    end

    prevsite = N + 1 #keep track of whether we are switching
    #to a new site to make sure F string
    #is only placed at most once for each site

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    perm = Vector{Int}(undef, Nt)
    sortperm!(perm, ITensors.terms(t); alg=InsertionSort, lt=isless_site)

    t = coefficient(t) * Prod(ITensors.terms(t)[perm])

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    parity = +1
    for n in Nt:-1:1
      currsite = ITensors.site(t[n])
      fermionic = has_fermion_string(
        ITensors.which_op(t[n]), only(sites[ITensors.site(t[n])])
      )
      if !ITensors.using_auto_fermion() && (parity == -1) && (currsite < prevsite)
        error("No verified fermion support for automatic TTN constructor!") # no verified support, just throw error
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        terms(t)[n] = Op("$(ITensors.which_op(t[n])) * F", only(ITensors.site(t[n])))
      end
      prevsite = currsite

      if fermionic
        error("No verified fermion support for automatic TTN constructor!") # no verified support, just throw error
        parity = -parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoTTN")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)
    # and account for anti-commuting, fermionic operators 
    # during above sort; put resulting sign into coef
    t *= ITensors.parity_sign(perm)
    ITensors.terms(os)[n] = t
  end
  return os
end

"""
    TTN(os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
    TTN(eltype::Type{<:Number}, os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
       
Convert an OpSum object `os` to a TreeTensorNetwork, with indices given by `sites`.
"""
function TTN(
  os::OpSum,
  sites::IndsNetwork{V,<:Index};
  root_vertex::V=default_root_vertex(sites),
  splitblocks=false,
  kwargs...,
)::TTN where {V}
  length(ITensors.terms(os)) == 0 && error("OpSum has no terms")
  is_tree(sites) || error("Site index graph must be a tree.")
  is_leaf(sites, root_vertex) || error("Tree root must be a leaf vertex.")

  os = deepcopy(os)
  os = sorteachterm(os, sites, root_vertex)
  os = ITensors.sortmergeterms(os) # not exported

  if hasqns(first(first(vertex_data(sites))))
    if !is_path_graph(sites)
      error(
        "OpSum → TTN constructor for QN conserving tensor networks only works for path/linear graphs right now.",
      )
    end
    # Use `ITensors.MPO` for now until general TTN constructor
    # works for QNs.
    # TODO: Check it is a path graph and get a linear arrangement!
    sites_linear_vertices = [only(sites[v]) for v in vertices(sites)]
    vertices_to_linear_vertices = Dictionary(vertices(sites), eachindex(vertices(sites)))
    os_linear_vertices = replace_vertices(os, vertices_to_linear_vertices)
    mpo = MPO(os_linear_vertices, sites_linear_vertices)
    tn = TTN(Dictionary(vertices(sites), [mpo[v] for v in 1:nv(sites)]))
    return tn
  end
  T = svdTTN(os, sites, root_vertex; kwargs...)
  if splitblocks
    error("splitblocks not yet implemented for AbstractTreeTensorNetwork.")
    T = ITensors.splitblocks(linkinds, T) # TODO: make this work
  end
  return T
end

function mpo(os::OpSum, external_inds::Vector; kwargs...)
  return TTN(os, path_indsnetwork(external_inds); kwargs...)
end

# Conversion from other formats
function TTN(o::Op, s::IndsNetwork; kwargs...)
  return TTN(OpSum{Float64}() + o, s; kwargs...)
end

function TTN(o::Scaled{C,Op}, s::IndsNetwork; kwargs...) where {C}
  return TTN(OpSum{C}() + o, s; kwargs...)
end

function TTN(o::Sum{Op}, s::IndsNetwork; kwargs...)
  return TTN(OpSum{Float64}() + o, s; kwargs...)
end

function TTN(o::Prod{Op}, s::IndsNetwork; kwargs...)
  return TTN(OpSum{Float64}() + o, s; kwargs...)
end

function TTN(o::Scaled{C,Prod{Op}}, s::IndsNetwork; kwargs...) where {C}
  return TTN(OpSum{C}() + o, s; kwargs...)
end

function TTN(o::Sum{Scaled{C,Op}}, s::IndsNetwork; kwargs...) where {C}
  return TTN(OpSum{C}() + o, s; kwargs...)
end

# Catch-all for leaf eltype specification
function TTN(eltype::Type{<:Number}, os, sites::IndsNetwork; kwargs...)
  return NDTensors.convert_scalartype(eltype, TTN(os, sites; kwargs...))
end


# 
# Tree adaptation of functionalities in ITensors.jl/src/physics/autompo/matelem.jl
# 

#################################
# ArrElem (simple sparse array) #
#################################

struct ArrElem{T,N}
  idxs::MVector{N,Int}
  val::T
end

function Base.:(==)(a1::ArrElem{T,N}, a2::ArrElem{T,N})::Bool where {T,N}
  return (a1.idxs == a2.idxs && a1.val == a2.val)
end

function Base.isless(a1::ArrElem{T,N}, a2::ArrElem{T,N})::Bool where {T,N}
  for n in 1:N
    if a1.idxs[n] != a2.idxs[n]
      return a1.idxs[n] < a2.idxs[n]
    end
  end
  return a1.val < a2.val
end


# 
# Sparse finite state machine construction
# 

# allow sparse arrays with ITensors.Sum entries
function Base.zero(::Type{S}) where {S<:Sum}
  return S()
end
Base.zero(t::Sum) = zero(typeof(t))

"""
    finite_state_machine(os::OpSum{C}, sites::IndsNetwork{V,<:Index}, root_vertex::V) where {C,V}

Finite state machine generator for ITensors.OpSum Hamiltonian defined on a tree graph. The
site Index graph must be a tree graph, and the chosen root  must be a leaf vertex of this
tree. Returns a DataGraph of SparseArrayKit.SparseArrays.
"""
function finite_state_machine(
  os::OpSum{C}, sites::IndsNetwork{V,<:Index}, root_vertex::V
) where {C,V}
  os = deepcopy(os)
  os = sorteachterm(os, sites, root_vertex)
  os = ITensors.sortmergeterms(os)

  ValType = ITensors.determineValType(ITensors.terms(os))

  # sparse symbolic representation of the TTN Hamiltonian as a DataGraph of SparseArrays
  sparseTTN = DataGraph{V,SparseArray{Sum{Scaled{ValType,Prod{Op}}}}}(
    underlying_graph(sites)
  )

  # traverse tree outwards from root vertex
  vs = reverse(post_order_dfs_vertices(sites, root_vertex))                                 # store vertices in fixed ordering relative to root
  es = reverse(reverse.(post_order_dfs_edges(sites, root_vertex)))                          # store edges in fixed ordering relative to root
  # some things to keep track of
  ranks = Dict(v => degree(sites, v) for v in vs)                                           # rank of every TTN tensor in network
  linkmaps = Dict(e => Dict{Prod{Op},Int}() for e in es)                                    # map from term in Hamiltonian to edge channel index for every edge
  site_coef_done = Prod{Op}[]                                                               # list of Hamiltonian terms for which the coefficient has been added to a site factor
  edge_orders = DataGraph{V,Vector{edgetype(sites)}}(underlying_graph(sites))               # relate indices of sparse TTN tensor to incident graph edges for each site

  for v in vs
    # collect all nontrivial entries of the TTN tensor at vertex v
    entries = Tuple{MVector{ranks[v],Int},Scaled{ValType,Prod{Op}}}[]

    # for every vertex, find all edges that contain this vertex
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    # use the corresponding ordering as index order for tensor elements at this site
    edge_orders[v] = edges
    dim_in = findfirst(e -> dst(e) == v, edges)
    edge_in = (isnothing(dim_in) ? [] : edges[dim_in])
    dims_out = findall(e -> src(e) == v, edges)
    edges_out = edges[dims_out]

    # sanity check, leaves only have single incoming or outgoing edge
    @assert !isempty(dims_out) || !isnothing(dim_in)
    (isempty(dims_out) || isnothing(dim_in)) && @assert is_leaf(sites, v)

    for term in os
      # loop over OpSum and pick out terms that act on current vertex
      crosses_vertex(term, sites, v) || continue

      # filter out factors that come in from the direction of the incoming edge
      incoming = filter(
        t -> edge_in ∈ edge_path(sites, ITensors.site(t), v), ITensors.terms(term)
      )
      # filter out factor that acts on current vertex
      onsite = filter(t -> (ITensors.site(t) == v), ITensors.terms(term))
      # for every outgoing edge, filter out factors that go out along that edge
      outgoing = Dict(
        e => filter(t -> e ∈ edge_path(sites, v, ITensors.site(t)), ITensors.terms(term))
        for e in edges_out
      )

      # translate into sparse tensor entry
      T_inds = MVector{ranks[v]}(fill(-1, ranks[v]))
      if !isnothing(dim_in) && !isempty(incoming)
        T_inds[dim_in] = ITensors.posInLink!(linkmaps[edge_in], ITensors.argument(term)) # get incoming channel
      end
      for dout in dims_out
        if !isempty(outgoing[edges[dout]])
          T_inds[dout] = ITensors.posInLink!(linkmaps[edges[dout]], ITensors.argument(term)) # add outgoing channel
        end
      end
      # if term starts at this site, add its coefficient as a site factor
      site_coef = one(C)
      if (isnothing(dim_in) || T_inds[dim_in] == -1) &&
        ITensors.argument(term) ∉ site_coef_done
        site_coef = ITensors.coefficient(term)
        push!(site_coef_done, ITensors.argument(term))
      end
      # add onsite identity for interactions passing through vertex
      if isempty(onsite)
        if !ITensors.using_auto_fermion() && isfermionic(incoming, sites)
          error("No verified fermion support for automatic TTN constructor!") # no verified support, just throw error
        else
          push!(onsite, Op("Id", v))
        end
      end
      # save indices and value of sparse tensor entry
      el = (T_inds, site_coef * Prod(onsite))
      push!(entries, el)
    end

    # handle start and end of operator terms and convert to sparse array
    linkdims = Tuple([
      (isempty(linkmaps[e]) ? 0 : maximum(values(linkmaps[e]))) + 2 for e in edges
    ])
    T = SparseArray{Sum{Scaled{ValType,Prod{Op}}},ranks[v]}(undef, linkdims)
    for (T_inds, t) in entries
      if !isnothing(dim_in)
        if T_inds[dim_in] == -1
          T_inds[dim_in] = 1 # always start in first channel
        else
          T_inds[dim_in] += 1 # shift regular channel
        end
      end
      if !isempty(dims_out)
        end_dims = filter(d -> T_inds[d] == -1, dims_out)
        normal_dims = filter(d -> T_inds[d] != -1, dims_out)
        T_inds[end_dims] .= linkdims[end_dims] # always end in last channel
        T_inds[normal_dims] .+= 1 # shift regular channels
      end
      T[T_inds...] += t
    end
    # add starting and ending identity operators
    if isnothing(dim_in)
      T[ones(Int, ranks[v])...] += one(ValType) * Prod([Op("Id", v)]) # only one real starting identity
    end
    # ending identities are a little more involved
    if !isnothing(dim_in)
      T[linkdims...] += one(ValType) * Prod([Op("Id", v)]) # place identity if all channels end
      # place identity from start of incoming channel to start of each single outgoing channel
      idT_end_inds = [linkdims...]
      idT_end_inds[dim_in] = 1
      for dout in dims_out
        idT_end_inds[dout] = 1
        T[idT_end_inds...] += one(ValType) * Prod([Op("Id", v)])
        idT_end_inds[dout] = linkdims[dout] # reset
      end
    end
    sparseTTN[v] = T
  end
  return sparseTTN, edge_orders
end

"""
    fsmTTN(os::OpSum{C}, sites::IndsNetwork{V,<:Index}, root_vertex::V, kwargs...) where {C,V}

Construct a dense TreeTensorNetwork from sparse finite state machine
represenatation, without compression.
"""
function fsmTTN(
  os::OpSum{C}, sites::IndsNetwork{V,<:Index}, root_vertex::V; trunc=false, kwargs...
)::TTN where {C,V}
  ValType = ITensors.determineValType(ITensors.terms(os))
  # start from finite state machine
  fsm, edge_orders = finite_state_machine(os, sites, root_vertex)
  # some trickery to get link dimension for every edge
  link_space = Dict{edgetype(sites),Index}()
  function get_linkind!(link_space, e)
    if !haskey(link_space, e)
      d = findfirst(x -> (x == e || x == reverse(e)), edge_orders[src(e)])
      link_space[e] = Index(size(fsm[src(e)], d), edge_tag(e))
    end
    return link_space[e]
  end
  # compress finite state machine into dense form
  H = TTN(sites)
  for v in vertices(sites)
    linkinds = [get_linkind!(link_space, e) for e in edge_orders[v]]
    linkdims = dim.(linkinds)
    H[v] = ITensor()
    for (T_ind, t) in nonzero_pairs(fsm[v])
      any(map(x -> abs(coefficient(x)) > eps(), t)) || continue
      T = zeros(ValType, linkdims...)
      T[T_ind] += one(ValType)
      T = itensor(T, linkinds)
      H[v] += T * computeSiteSum(sites, t)
    end
  end
  # add option for numerical truncation, but throw warning as this can fail sometimes
  if trunc
    @warn "Naive numerical truncation of TTN Hamiltonian may fail for larger systems."
    # see https://github.com/ITensor/ITensors.jl/issues/526
    lognormT = lognorm(H)
    H /= exp(lognormT / nv(H)) # TODO: fix broadcasting for in-place assignment
    H = truncate(H; root_vertex, kwargs...)
    H *= exp(lognormT / nv(H))
  end
  return H
end

function computeSiteSum(
  sites::IndsNetwork{V,<:Index}, ops::Sum{Scaled{C,Prod{Op}}}
)::ITensor where {V,C}
  ValType = ITensors.determineValType(ITensors.terms(ops))
  v = ITensors.site(ITensors.argument(ops[1])[1])
  T =
    convert(ValType, coefficient(ops[1])) *
    computeSiteProd(sites, ITensors.argument(ops[1]))
  for j in 2:length(ops)
    (ITensors.site(ITensors.argument(ops[j])[1]) != v) &&
      error("Mismatch of vertex labels in computeSiteSum")
    T +=
      convert(ValType, coefficient(ops[j])) *
      computeSiteProd(sites, ITensors.argument(ops[j]))
  end
  return T
end
