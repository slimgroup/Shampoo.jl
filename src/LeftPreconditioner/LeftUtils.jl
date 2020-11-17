export hamm_op, integral_shot

function hamm_op(d_obs::judiVector{vDT},offset::Array) where {vDT}
    d_out = deepcopy(d_obs)
    nsrc, nrec = size(offset)
    max_offset = maximum(offset)
    hamm_weights = convert(Array{vDT},0.54 .- 0.46*cos.(2*pi*offset/max_offset))
    for i = 1:nsrc
        d_out.data[i] = d_obs.data[i] .* sqrt.(hamm_weights[i,:]')
    end
    return d_out
end

function integral_shot(C::joLinearFunction{ADDT,ARDT},d_obs::judiVector{avDT}) where {ADDT,ARDT,avDT}
    # fractional integration based on C
    C.n == size(d_obs.data[1],1) || throw(judiVectorException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",C.name,typeof(C),avDT]," / "))
	d_out = deepcopy(d_obs)
	nsrc = d_obs.nsrc
	for i = 1:nsrc
		d_out.data[i] = C*d_obs.data[i]
	end
	return d_out
end