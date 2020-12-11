export  integral_shot

function apply_frac_integral(C::joLinearFunction{ADDT,ARDT},d_obs::judiVector{avDT}) where {ADDT,ARDT,avDT}
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