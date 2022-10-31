"""
Relabel sites in OpSum according to given site map
"""
function relabel_sites(O::OpSum, vmap::AbstractDictionary)
  Oout = OpSum()
  for term in Ops.terms(O)
    c = Ops.coefficient(term)
    p = Ops.argument(term)
    # swap sites for every Op in product and multiply resulting Ops
    pout = prod([
      Op(Ops.which_op(o), map(v -> vmap[v], Ops.sites(o))...; Ops.params(o)...) for o in p
    ])
    # add to new OpSum
    Oout += c * pout
  end
  return Oout
end
