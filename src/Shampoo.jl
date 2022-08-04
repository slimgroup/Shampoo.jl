__precompile__()
module Shampoo

    using Requires
    using JOLI, LinearAlgebra, FFTW
    export FractionalIntegrationOp

    function __init__()
        @require JUDI="f3b833dc-6b2e-5b9c-b940-873ed6319979" begin
            using .JUDI
            function FractionalIntegrationOp(J::judiJacobian, order::Number=.5f0)
                return FractionalIntegrationOp(J.F.rInterpolation.geometry.nt[1],
                J.F.rInterpolation.geometry.dt[1],
                J.q.nsrc,
                length(J.F.rInterpolation.geometry.xloc[1]), order)
            end
        end
    end

    # left preconditioners: work in data domain
    function FractionalIntegrationOp(nt::Int, dt::Float32, nsrc::Int, nrec::Int, order::Number=.5f0)

	    ## Follows https://github.com/SINBADconsortium/SLIM-release-apps/blob/master/tools/algorithms/TimeModeling/opHalfInt.m

        F = joDFT(nt; DDT=Float32, RDT=ComplexF32, centered=true);

        df = 1f0/dt;
        freq = -(nt-1)/2*df:df:(nt-1)/2*df;
        omega = freq*2*pi;
        partialDt = (abs.(omega)).^(-order);
	    order>0 && (partialDt[Int(ceil((nt-1)/2))+1] = 1f0);    # avoid division by 0

        P1 = F'*joDiag(ComplexF32.(partialDt))*F;               # preconditioner for a single trace
        return joKron(joEye(nsrc, DDT=Float32, RDT=Float32), joEye(nrec; DDT=Float32, RDT=Float32), P1);

    end

end