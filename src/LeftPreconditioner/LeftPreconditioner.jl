# Integrations on shot records
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

export LeftPrecondJ

###############################################################################
# Left precontioners for system P1*J*dm = P1*(F*q-F0*q)
# where P1 is fractional integration

function LeftPrecondJ(J::judiJacobian)
	
	nt = J.recGeometry.nt[1]
	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{Float32,2},fgl_deriv(0.5,x,1));
	C    = joConvolve(nt,1,y[:]);
	return C
end
