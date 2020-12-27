__precompile__()
module SeismicPreconditioners

using JUDI.TimeModeling, JOLI, LinearAlgebra, FFTW, SpecialFunctions, DSP

export CumsumOp,DiffOp,HammingOp,FractionalIntegrationOp,FractionalLaplacianOp, MaskOp

# utilities
include("utils/adj_diff_cumsum.jl")
include("utils/auxilary.jl")
include("utils/fgl_deriv.jl")
include("utils/Filter_freq_band.jl")
include("utils/joConvolve.jl")
include("utils/wavelet.jl")

# operators
include("operators/fractional_integration.jl")
include("operators/fractional_laplacian.jl")
include("operators/hamming_operator.jl")
include("operators/mask_op.jl")

# left preconditioners: work in data domain

function CumsumOp(nt::Int,nsrc::Int,nrec::Int;DDT=Float32)
	P = joLinearFunctionFwd_T(nt*nsrc*nrec, nt*nsrc*nrec,
	                             v -> cumsum(v),
	                             w -> adjoint_cumsum(w),
								 DDT,DDT,name="cumsum operator")
end

function DiffOp(nt::Int,nsrc::Int,nrec::Int;order=1,DDT=Float32)
	P = joLinearFunctionFwd_T(nt*nsrc*nrec, nt*nsrc*nrec,
	                             v -> diff(v),
	                             w -> adjoint_diff(w),
								 DDT,DDT,name="diff operator")
	return P
end

function HammingOp(nt::Int,nsrc::Int,nrec::Int;DDT=Float32)
	H = joLinearFunctionFwd_T(nt*nsrc*nrec, nt*nsrc*nrec,
								v -> hamm_op(v),
								w -> hamm_op(w),
								DDT,DDT,name="Hamming operator")
	return H
end

function FractionalIntegrationOp(nt::Int,nsrc::Int,nrec::Int,order::Number;DDT=Float32)
	### Positive order -> integration, Negative order -> derivative
	C = frac_int(nt,order;DDT=DDT)
	P = joLinearFunctionFwd_T(nt*nsrc*nrec, nt*nsrc*nrec,
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 DDT,DDT,name="Fractional integration operator")
	return P
end

# right preconditioners: work in model domain

function MaskOp(n::Tuple,mask::Array{DDT}) where {DDT}
	P = joLinearFunctionFwd_T(prod(n), prod(n),
								v -> mask_op(v,mask),
								w -> mask_op(w,mask),
								DDT,DDT,name="Mask operator")
	return P
end

function FractionalLaplacianOp(F::judiPDEextended,order::Number)
	model = F.model
	info = F.info
	P = laplacian_operator(model::Model,order::Number,info::Info)
	return P
end

function FractionalLaplacianOp(J::judiJacobian,order::Number)
	model = J.model
    info = J.info
    info1 = deepcopy(info)
	info1.nsrc = 1
	P = laplacian_operator(model::Model,order::Number,info1::Info)
	return P
end

end