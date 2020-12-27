export hamm_op

function hamm_op(d_obs::judiVector{vDT}) where {vDT}
    d_out = deepcopy(d_obs)
    nrec = length(d_obs.geometry.xloc[1])
    hamm_weights = convert(Array{vDT},reshape(hamming(nrec), 1, :))
    for i = 1:d_obs.nsrc
        d_out.data[i] = d_obs.data[i] .* (hamm_weights)
    end
    return d_out
end
