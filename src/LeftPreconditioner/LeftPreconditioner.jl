# Integrations on shot records
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

export LeftPrecondJ

###############################################################################
# Left precontioners for system P1*J*dm = P1*(F*q-F0*q)
# where P1 is fractional integration

function LeftPrecondJ(J::judiJacobian)
	nsrc = length(J.srcGeometry.xloc)
	nt = J.recGeometry.nt[1]

	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{Float32,2},fgl_deriv(0.5,x,1));	# half integration
	C    = joConvolve(nt,1,y[:];DDT=Float32,RDT=Float32);

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> integral_shot(C,v),
	                             w -> integral_shot(C',w),
								 Float32,Float32,name="Fractional integration operator")
end

function integral_shot(C::joLinearFunction,d_obs::judiVector)
	d_out = deepcopy(d_obs)
	nrec = length(d_obs.geometry.xloc[1])
	nt = d_obs.geometry.nt[1]
	nsrc = d_obs.nsrc
	for i = 1:nsrc
		for j = 1:nrec
			d_out.data[i][:,j] = C*d_lin.data[i][:,j]
		end
	end
	return d_out
end

