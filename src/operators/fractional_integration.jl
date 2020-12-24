# Fractional Integration -- 
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2020

export apply_frac_integral, frac_int

###############################################################################
# Fractional integration

function frac_int(nt::Int,order::Number;DDT=Float32)
	# fractional integration for a trace
	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{DDT,2},fgl_deriv(-order,x,1));	# fractional integration
	C    = joConvolve(nt,1,y[:];DDT=DDT,RDT=DDT);
	return C
end

function apply_frac_integral(C::joLinearFunction{ADDT,ARDT},d_obs::judiVector{avDT}) where {ADDT,ARDT,avDT}
    # fractional integration based on C
    C.n == size(d_obs.data[1],1) || throw(judiVectorException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",C.name,typeof(C),avDT]," / "))
	d_out = deepcopy(d_obs)
	for i = 1:d_obs.nsrc
		d_out.data[i] = C*d_obs.data[i]
	end
	return d_out
end