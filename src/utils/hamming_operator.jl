export hamm_op

function hamm_op(d_obs::judiVector{vDT},offset::Array) where {vDT}
    d_out = deepcopy(d_obs)
    nsrc, nrec = size(offset)
    hamm_weights = convert(Array{vDT},reshape(hamming(nrec), 1, :))
    for i = 1:nsrc
        d_out.data[i] = d_obs.data[i] .* sqrt.(hamm_weights)
    end
    return d_out
end
