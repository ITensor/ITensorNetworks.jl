using Graphs: degree, is_tree
using ITensors: flux, has_fermion_string, itensor, ops, removeqns, space, val
using ITensors.ITensorMPS: ITensorMPS, cutoff, linkdims, truncate!
using ITensors.LazyApply: Prod, Sum, coefficient
using ITensors.NDTensors: Block, blockdim, maxdim, nblocks, nnzblocks
using ITensors.Ops: argument, coefficient, Op, OpSum, name, params, site, terms, which_op
using NamedGraphs.GraphsExtensions:
  GraphsExtensions, boundary_edges, degrees, is_leaf_vertex, vertex_path
using StaticArrays: MVector

# 
# Utility methods
# 

function align_edges(edges, reference_edges)
  return intersect(Iterators.flatten(zip(edges, reverse.(edges))), reference_edges)
end

function align_and_reorder_edges(edges, reference_edges)
  return intersect(reference_edges, align_edges(edges, reference_edges))
end

function split_at_vertex(g::AbstractGraph, v)
  g = copy(g)
  rem_vertex!(g, v)
  return Set.(connected_components(g))
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo.jl
# 

"""
    ttn_svd(os::OpSum, sites::IndsNetwork, root_vertex, kwargs...)

Construct a TreeTensorNetwork from a symbolic OpSum representation of a
Hamiltonian, compressing shared interaction channels.
"""
function ttn_svd(os::OpSum, sites::IndsNetwork, root_vertex; kwargs...)
  # Function barrier to improve type stability
  coefficient_type = ITensorMPS.determineValType(terms(os))
  return ttn_svd(coefficient_type, os, sites, root_vertex; kwargs...)
end

function ttn_svd(
  coefficient_type::Type{<:Number},
  os::OpSum,
  sites0::IndsNetwork,
  root_vertex;
  mindim::Int=1,
  maxdim::Int=typemax(Int),
  cutoff=eps(real(coefficient_type)) * 10,
)
  linkdir_ref = ITensors.In   # safe to always use autofermion default here

  sites = copy(sites0)  # copy because of modification to handle internal indices 
  edgetype_sites = edgetype(sites)
  vertextype_sites = vertextype(sites)
  thishasqns = any(v -> hasqns(sites[v]), vertices(sites))

  # traverse tree outwards from root vertex
  vs = _default_vertex_ordering(sites, root_vertex)
  vert_number = Dict(zip(vs, 1:length(vs)))
  # TODO: Add check in ttn_svd that the ordering matches that of find_index_in_tree, which is used in sorteachterm #fermion-sign!
  # store edges in fixed ordering relative to root
  ordered_edges = _default_edge_ordering(sites, root_vertex)
  # some things to keep track of
  # rank of every TTN tensor in network
  degrees = Dict(v => degree(sites, v) for v in vs)
  # link isometries for SVD compression of TTN
  Vs = Dict(e => Dict{QN,Matrix{coefficient_type}}() for e in ordered_edges)
  # map from term in Hamiltonian to incoming channel index for every edge
  inmaps = Dict{Pair{edgetype_sites,QN},Dict{Vector{Op},Int}}()
  # map from term in Hamiltonian to outgoing channel index for every edge
  outmaps = Dict{Pair{edgetype_sites,QN},Dict{Vector{Op},Int}}()

  op_cache = Dict{Pair{String,vertextype_sites},ITensor}()

  function calc_qn(term::Vector{Op})
    q = QN()
    for st in term
      op_tensor = get(op_cache, ITensors.which_op(st) => ITensors.site(st), nothing)
      if op_tensor === nothing
        op_tensor = op(
          sites[ITensors.site(st)], ITensors.which_op(st); ITensors.params(st)...
        )
        op_cache[ITensors.which_op(st) => ITensors.site(st)] = op_tensor
      end
      if !isnothing(flux(op_tensor))
        q += flux(op_tensor)
      end
    end
    return q
  end

  Hflux = -calc_qn(terms(first(terms(os))))

  # insert dummy indices on internal vertices, these will not show up in the final tensor
  is_internal = Dict{vertextype_sites,Bool}()
  for v in vs
    is_internal[v] = isempty(sites[v])
    if isempty(sites[v])
      # FIXME: This logic only works for trivial flux, breaks for nonzero flux
      # TODO: add assert or fix and add test!
      sites[v] = [Index(Hflux => 1)]
    end
  end

  # Bond coefficients for incoming edge channels.
  # These become the "M" coefficient matrices that get SVD'd.
  inbond_coefs = Dict(
    e => Dict{QN,Vector{ITensorMPS.MatElem{coefficient_type}}}() for e in ordered_edges
  )
  # list of terms for which the coefficient has been added to a site factor
  site_coef_done = Prod{Op}[]
  # Temporary symbolic representation of TTN Hamiltonian
  tempTTN = Dict(v => QNArrElem{Scaled{coefficient_type,Prod{Op}},degrees[v]}[] for v in vs)

  # Build compressed finite state machine representation (tempTTN)
  for v in vs
    # For every vertex, find all edges that contain this vertex
    edges = align_and_reorder_edges(incident_edges(sites, v), ordered_edges)

    # Use the corresponding ordering as index order for tensor elements at this site
    dim_in = findfirst(e -> dst(e) == v, edges)
    edge_in = (isnothing(dim_in) ? [] : edges[dim_in])
    dims_out = findall(e -> src(e) == v, edges)
    edges_out = edges[dims_out]

    # For every site w except v, determine the incident edge to v that lies 
    # in the edge_path(w,v)
    subgraphs = split_at_vertex(sites, v)
    _boundary_edges = align_edges(
      [only(boundary_edges(underlying_graph(sites), subgraph)) for subgraph in subgraphs],
      edges,
    )
    which_incident_edge = Dict(
      Iterators.flatten([
        subgraphs[i] .=> ((_boundary_edges[i]),) for i in eachindex(subgraphs)
      ]),
    )

    # sanity check, leaves only have single incoming or outgoing edge
    @assert !isempty(dims_out) || !isnothing(dim_in)
    (isempty(dims_out) || isnothing(dim_in)) && @assert is_leaf_vertex(sites, v)

    for term in os
      # Loop over OpSum and pick out terms that act on current vertex
      ops = ITensors.terms(term)
      if v in ITensors.site.(ops)
        crosses_vertex = true
      else
        crosses_vertex =
          !isone(
            length(Set([which_incident_edge[site] for site in ITensors.site.(ops)]))
          )
      end
      # If term doesn't cross vertex, skip it
      crosses_vertex || continue


      # filter out factor that acts on current vertex
      onsite = filter(t -> (ITensors.site(t) == v), ops)
      not_onsite_ops = setdiff(ops, onsite)

      # filter out ops that come in from the direction of the incoming edge
      incoming = filter(
        t -> which_incident_edge[ITensors.site(t)] == edge_in, not_onsite_ops
      )

      # also store all non-incoming ops in standard order, used for channel merging
      not_incoming = filter(
        t -> (ITensors.site(t) == v) || which_incident_edge[ITensors.site(t)] != edge_in,
        ops,
      )

      # for every outgoing edge, filter out ops that go out along that edge
      outgoing = Dict(
        e => filter(t -> which_incident_edge[ITensors.site(t)] == e, not_onsite_ops) for
        e in edges_out
      )

      # compute QNs
      incoming_qn = calc_qn(incoming)
      not_incoming_qn = calc_qn(not_incoming)
      outgoing_qns = Dict(e => calc_qn(outgoing[e]) for e in edges_out)
      site_qn = calc_qn(onsite)

      # initialize QNArrayElement indices and quantum numbers 
      T_inds = MVector{degrees[v]}(fill(-1, degrees[v]))
      T_qns = MVector{degrees[v]}(fill(QN(), degrees[v]))
      # initialize ArrayElement indices for inbond_coefs
      bond_row = -1
      bond_col = -1
      if !isempty(incoming)
        # get the correct map from edge=>QN to term and channel
        # this checks if term exists on edge=>QN ( otherwise insert it) and returns it's index
        coutmap = get!(outmaps, edge_in => not_incoming_qn, Dict{Vector{Op},Int}())
        cinmap = get!(inmaps, edge_in => -incoming_qn, Dict{Vector{Op},Int}())

        bond_row = ITensorMPS.posInLink!(cinmap, incoming)
        bond_col = ITensorMPS.posInLink!(coutmap, not_incoming) # get incoming channel
        bond_coef = convert(coefficient_type, coefficient(term))
        q_inbond_coefs = get!(
          inbond_coefs[edge_in], incoming_qn, ITensorMPS.MatElem{coefficient_type}[]
        )
        push!(q_inbond_coefs, ITensorMPS.MatElem(bond_row, bond_col, bond_coef))
        T_inds[dim_in] = bond_col
        T_qns[dim_in] = -incoming_qn
      end
      for dout in dims_out
        coutmap = get!(
          outmaps, edges[dout] => outgoing_qns[edges[dout]], Dict{Vector{Op},Int}()
        )
        # add outgoing channel
        T_inds[dout] = ITensorMPS.posInLink!(coutmap, outgoing[edges[dout]])
        T_qns[dout] = outgoing_qns[edges[dout]]
      end
      # if term starts at this site, add its coefficient as a site factor
      site_coef = one(coefficient_type)
      if (isnothing(dim_in) || T_inds[dim_in] == -1) &&
        argument(term) ∉ site_coef_done
        site_coef = coefficient(term)
        # required since coefficient seems to return ComplexF64 even if coefficient_type is determined to be real
        site_coef = convert(coefficient_type, site_coef)
        push!(site_coef_done, argument(term))
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
      el = QNArrElem(T_qns, T_inds, site_coef * Prod(onsite))

      push!(tempTTN[v], el)
    end

    ITensorMPS.remove_dups!(tempTTN[v])
    # manual truncation: isometry on incoming edge
    if !isnothing(dim_in) && !isempty(inbond_coefs[edges[dim_in]])
      for (q, mat) in inbond_coefs[edges[dim_in]]
        M = ITensorMPS.toMatrix(mat)
        U, S, V = svd(M)
        P = S .^ 2
        truncate!(P; maxdim, cutoff, mindim)
        tdim = length(P)
        nc = size(M, 2)
        Vs[edges[dim_in]][q] = Matrix{coefficient_type}(V[1:nc, 1:tdim])
      end
    end
  end

  # compress this tempTTN representation into dense form

  link_space = Dict{edgetype_sites,Index}()
  for e in ordered_edges
    operator_blocks = [q=>size(Vq,2) for (q,Vq) in Vs[e]]
    link_space[e] = Index(QN()=>1,operator_blocks...,Hflux=>1; tags=edge_tag(e), dir=linkdir_ref)
  end

  H = ttn(sites0)   # initialize TTN without the dummy indices added
  function qnblock(i::Index, q::QN)
    for b in 2:(nblocks(i) - 1)
      flux(i, Block(b)) == q && return b
    end
    return error("Could not find block of QNIndex with matching QN")
  end
  qnblockdim(i::Index, q::QN) = blockdim(i, qnblock(i, q))

  for v in vs
    # redo the whole thing like before
    # TODO: use neighborhood instead of going through all edges, see above
    edges = align_and_reorder_edges(incident_edges(sites, v), ordered_edges)
    dim_in = findfirst(e -> dst(e) == v, edges)
    dims_out = findall(e -> src(e) == v, edges)
    # slice isometries at this vertex
    Vv = [Vs[e] for e in edges]
    linkinds = [link_space[e] for e in edges]

    # construct blocks
    blocks = Dict{Tuple{Block{degrees[v]},Vector{Op}},Array{coefficient_type,degrees[v]}}()
    for el in tempTTN[v]
      t = el.val
      (abs(coefficient(t)) > eps(real(coefficient_type))) || continue
      block_helper_inds = fill(-1, degrees[v]) # we manipulate T_inds later, and loose track of ending/starting information, so keep track of it here
      T_inds = el.idxs
      T_qns = el.qn_idxs
      ct = convert(coefficient_type, coefficient(t))
      sublinkdims = [
        (T_inds[i] == -1 ? 1 : qnblockdim(linkinds[i], T_qns[i])) for i in 1:degrees[v]
      ]
      zero_arr() = zeros(coefficient_type, sublinkdims...)
      terminal_dims = findall(d -> T_inds[d] == -1, 1:degrees[v])   # directions in which term starts or ends
      normal_dims = findall(d -> T_inds[d] ≠ -1, 1:degrees[v])      # normal dimensions, do truncation thingies
      T_inds[terminal_dims] .= 1                                  # start in channel 1  ###??
      block_helper_inds[terminal_dims] .= 1
      for dout in filter(d -> d ∈ terminal_dims, dims_out)
        T_inds[dout] = sublinkdims[dout]                          # end in channel linkdims[d] for each dimension d
        @assert isone(T_inds[dout])
        block_helper_inds[dout] = nblocks(linkinds[dout])
      end

      # set non-trivial helper inds
      for d in normal_dims
        block_helper_inds[d] = qnblock(linkinds[d], T_qns[d])
      end
      @assert all(≠(-1), block_helper_inds)# check that all block indices are set

      # make and fill Block 
      theblock = Block(Tuple(block_helper_inds))
      if isempty(normal_dims)
        M = get!(blocks, (theblock, terms(t)), zero_arr())
        @assert isone(length(M))
        M[] += ct
      else
        M = get!(blocks, (theblock, terms(t)), zero_arr())
        dim_ranges = Tuple(size(Vv[d][T_qns[d]], 2) for d in normal_dims)
        for c in CartesianIndices(dim_ranges) # applies isometries in a element-wise manner
          z = ct
          temp_inds = copy(T_inds)
          for (i, d) in enumerate(normal_dims)
            V_factor = Vv[d][T_qns[d]][T_inds[d], c[i]]
            z *= (d == dim_in ? conj(V_factor) : V_factor) # conjugate incoming isometry factor
            temp_inds[d] = c[i]
          end
          M[temp_inds...] += z
        end
      end
    end

    H[v] = ITensor()

    # Set the final arrow directions
    if !isnothing(dim_in)
      linkinds[dim_in] = dag(linkinds[dim_in])
    end

    for ((b, q_op), m) in blocks
      Op = computeSiteProd(sites, Prod(q_op))
      if hasqns(Op)
        # FIXME: this may not be safe, we may want to check for the equivalent (zero tensor?) case in the dense case as well
        iszero(nnzblocks(Op)) && continue
      end
      sq = flux(Op)
      if !isnothing(sq)
        rq = (b[1] == 1 ? Hflux : first(space(linkinds[1])[b[1]])) # get row (dim_in) QN
        cq = rq - sq # get column (out_dims) QN
        if ITensors.using_auto_fermion()
          # we need to account for the direct product below ordering the physical indices as the last indices
          # although they are in between incoming and outgoing indices in the canonical site-ordering
          perm = (1, 3, 2)
          if ITensors.compute_permfactor(perm, rq, sq, cq) == -1
            Op .*= -1
          end
        end
      end
      T = ITensors.BlockSparseTensor(coefficient_type, [b], linkinds)
      T[b] .= m
      iT = itensor(T)
      if !thishasqns
        iT = removeqns(iT)
      end

      if is_internal[v]
        H[v] += iT
      else
        #TODO: Remove this assert since it seems to be costly
        #if hasqns(iT)
        #  @assert flux(iT * Op) == Hflux
        #end
        H[v] += (iT * Op)
      end
    end

    linkdims = dim.(linkinds)
    # add starting and ending identity operators
    idT = zeros(coefficient_type, linkdims...)
    if isnothing(dim_in)
      # only one real starting identity
      idT[ones(Int, degrees[v])...] = 1.0
    end
    # ending identities are a little more involved
    if !isnothing(dim_in)
      # place identity if all channels end
      idT[linkdims...] = 1.0
      # place identity from start of incoming channel to start of each single outgoing channel, and end all other channels
      idT_end_inds = [linkdims...]
      #this should really be an int
      idT_end_inds[dim_in] = 1
      for dout in dims_out
        idT_end_inds[dout] = 1
        idT[idT_end_inds...] = 1.0
        # reset
        idT_end_inds[dout] = linkdims[dout]
      end
    end

    T = itensor(idT, linkinds)
    if !thishasqns
      T = removeqns(T)
    end
    if is_internal[v]
      H[v] += T
    else
      H[v] += T * ITensorNetworks.computeSiteProd(sites, Prod([(Op("Id", v))]))
    end
  end

  return H
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo_generic.jl
# 

# TODO: fix fermion support, definitely broken

# needed an extra `only` compared to ITensors version since IndsNetwork has Vector{<:Index}
# as vertex data
function isfermionic(t::Vector{Op}, sites::IndsNetwork{V,<:Index}) where {V}
  p = +1
  for op in t
    if has_fermion_string(name(op), only(sites[site(op)]))
      p *= -1
    end
  end
  return (p == -1)
end

# only(site(ops[1])) in ITensors breaks for Tuple site labels, had to drop the only
function computeSiteProd(sites::IndsNetwork{V,<:Index}, ops::Prod{Op})::ITensor where {V}
  v = site(ops[1])
  T = op(sites[v], which_op(ops[1]); params(ops[1])...)
  for j in 2:length(ops)
    (site(ops[j]) != v) && error("Mismatch of vertex labels in computeSiteProd")
    opj = op(sites[v], which_op(ops[j]); params(ops[j])...)
    T = product(T, opj)
  end
  return T
end

function _default_vertex_ordering(g::AbstractGraph, root_vertex)
  return reverse(post_order_dfs_vertices(g, root_vertex))
end

function _default_edge_ordering(g::AbstractGraph, root_vertex)
  return reverse(reverse.(post_order_dfs_edges(g, root_vertex)))
end

function check_terms_support(os::OpSum, sites)
  for t in os
    if !all(map(v -> has_vertex(sites, v), ITensors.sites(t)))
      error(
        "The OpSum contains a term $t that does not have support on the underlying graph."
      )
    end
  end
end

# This code is very similar to ITensorMPS sorteachterm in opsum_generic.jl
function sorteachterm(os::OpSum, sites, root_vertex)
  os = copy(os)

  # Build the isless_site function to pass to sortperm below:
  # + ordering = array of vertices ordered relative to chosen root, chosen outward from root
  # + site_positions = map from vertex to where it is in ordering (inverse map of `ordering`)
  ordering = _default_vertex_ordering(sites, root_vertex)
  site_positions = Dict(zip(ordering, 1:length(ordering)))
  findpos(op::Op) = site_positions[site(op)]
  isless_site(o1::Op, o2::Op) = findpos(o1) < findpos(o2)

  N = nv(sites)
  for j in eachindex(os)
    t = os[j]

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    Nt = length(t)
    #perm = Vector{Int}(undef, Nt)
    perm = sortperm(terms(t); alg=InsertionSort, lt=isless_site)
    t = coefficient(t) * Prod(terms(t)[perm])

    # Everything below deals with fermionic operators:

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    prevsite = typemax(Int) #keep track of whether we are switching to a new site
    t_parity = +1
    for n in reverse(1:Nt)
      currsite = site(t[n])
      fermionic = has_fermion_string(
        which_op(t[n]), only(sites[site(t[n])])
      )
      if !ITensors.using_auto_fermion() && (t_parity == -1) && (currsite < prevsite)
        error("No verified fermion support for automatic TTN constructor!") # no verified support, just throw error
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        terms(t)[n] = Op("$(which_op(t[n])) * F", only(site(t[n])))
      end
      prevsite = currsite

      if fermionic
        t_parity = -t_parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if t_parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoTTN")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)
    # and account for anti-commuting, fermionic operators 
    # during above sort; put resulting sign into coef
    t *= ITensors.parity_sign(perm)
    terms(os)[j] = t
  end
  return os
end

"""
    ttn(os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
    ttn(eltype::Type{<:Number}, os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
       
Convert an OpSum object `os` to a TreeTensorNetwork, with indices given by `sites`.
"""
function ttn(
  os::OpSum,
  sites::IndsNetwork;
  root_vertex=GraphsExtensions.default_root_vertex(sites),
  kwargs...,
)
  length(terms(os)) == 0 && error("OpSum has no terms")
  is_tree(sites) || error("Site index graph must be a tree.")
  is_leaf_vertex(sites, root_vertex) || error("Tree root must be a leaf vertex.")
  check_terms_support(os,sites)
  os = deepcopy(os) #TODO: do we need this? sorteachterm copies `os` again
  os = sorteachterm(os, sites, root_vertex)
  os = ITensorMPS.sortmergeterms(os)
  return ttn_svd(os, sites, root_vertex; kwargs...)
end

function mpo(os::OpSum, external_inds::Vector; kwargs...)
  return ttn(os, path_indsnetwork(external_inds); kwargs...)
end

# Conversion from other formats
function ttn(o::Op, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Scaled{C,Op}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

function ttn(o::Sum{Op}, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Prod{Op}, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Scaled{C,Prod{Op}}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

function ttn(o::Sum{Scaled{C,Op}}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

# Catch-all for leaf eltype specification
function ttn(eltype::Type{<:Number}, os, sites::IndsNetwork; kwargs...)
  return NDTensors.convert_scalartype(eltype, ttn(os, sites; kwargs...))
end

#####################################
# QNArrElem (sparse array with QNs) #
#####################################

struct QNArrElem{T,N}
  qn_idxs::MVector{N,QN}
  idxs::MVector{N,Int}
  val::T
end

function Base.:(==)(a1::QNArrElem{T,N}, a2::QNArrElem{T,N})::Bool where {T,N}
  return (a1.idxs == a2.idxs && a1.val == a2.val && a1.qn_idxs == a2.qn_idxs)
end

function Base.isless(a1::QNArrElem{T,N}, a2::QNArrElem{T,N})::Bool where {T,N}
  ###two separate loops s.t. the MPS case reduces to the ITensors Implementation of QNMatElem
  for n in 1:N
    if a1.qn_idxs[n] != a2.qn_idxs[n]
      return a1.qn_idxs[n] < a2.qn_idxs[n]
    end
  end
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
