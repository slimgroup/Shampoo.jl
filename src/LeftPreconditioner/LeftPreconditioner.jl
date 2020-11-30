# Integrations on shot records
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

export LeftPrecondJ, HammingPrecond

###############################################################################
# Left precontioners for system P1*J*dm = P1*(F*q-F0*q)
# where P1 is fractional integration

function LeftPrecondJ(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nsrc = length(J.srcGeometry.xloc)
	nt = J.recGeometry.nt[1]

	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{ARDT,2},fgl_deriv(-0.5,x,1));	# half integration
	C    = joConvolve(nt,1,y[:];DDT=ARDT,RDT=ARDT);

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> integral_shot(C,v),
	                             w -> integral_shot(C',w),
								 ARDT,ARDT,name="Fractional integration operator")
end

function CumsumPrecondJ(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nsrc = length(J.srcGeometry.xloc)
	nt = J.recGeometry.nt[1]

	C = joLinearFunctionFwd_T(nt, nt,
		v -> cumsum(v,dims=1),
		w -> reverse(cumsum(reverse(v,dims=1),dims=1),dims=1),
		ARDT,ARDT,name="cumsum operator")

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> integral_shot(C,v),
	                             w -> integral_shot(C',w),
								 ARDT,ARDT,name="Fractional integration operator")
end

function HammingPrecond(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nsrc = length(J.srcGeometry.xloc)
	nrec = length(J.recGeometry.xloc[1])
	offset = zeros(Float32,nsrc,nrec)
	for i = 1:nsrc
		offset[i,:] = abs.(J.srcGeometry.xloc[i] .- J.recGeometry.xloc[i])
	end
	H = joLinearFunctionFwd_T(size(J,1), size(J,1),
								v -> hamm_op(v,offset),
								w -> hamm_op(w,offset),
								ARDT,ARDT,name="Hamming operator")
end

