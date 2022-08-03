__precompile__()
module Shampoo

using JOLI, LinearAlgebra, FFTW
export FractionalIntegrationOp

# left preconditioners: work in data domain

function FractionalIntegrationOp(nt::Int,dt::Float32,nsrc::Int,nrec::Int,order::Number=.5f0)

	## Follows https://github.com/SINBADconsortium/SLIM-release-apps/blob/master/tools/algorithms/TimeModeling/opHalfInt.m

    F = joDFT(nt; DDT=Float32, RDT=ComplexF32, centered=true);

    df = 1f0/dt;
    freq = -(nt-1)/2*df:df:(nt-1)/2*df;
    omega = freq*2*pi;
    partialDt = (abs.(omega)).^(-order);
	order>0 && (partialDt[Int(ceil((nt-1)/2))+1] = 1f0);
    P1 = F'*joDiag(ComplexF32.(partialDt))*F;

    P = joKron(joEye(nsrc, DDT=Float32, RDT=Float32), joEye(nrec; DDT=Float32, RDT=Float32), P1);
	
	return P

end

end