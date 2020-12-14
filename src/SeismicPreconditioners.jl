__precompile__()
module SeismicPreconditioners

using JUDI.TimeModeling, JOLI, LinearAlgebra, FFTW, SpecialFunctions, DSP

export CumsumOp,DiffOp,HammingOp,FractionalIntegrationOp,FractionalLaplacianOp

# utilities
include("utils/apply_frac_integral.jl")
include("utils/Filter_freq_band.jl")
include("utils/fractional_integration.jl")
include("utils/fractional_laplacian.jl")
include("utils/hamming_operator.jl")
include("utils/joConvolve.jl")
include("utils/wavelet.jl")

# left preconditioners: work in data domain

function CumsumOp(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nt = J.recGeometry.nt[1]

	C = joLinearFunctionFwd_T(nt, nt,
		v -> cumsum(v,dims=1),
		w -> reverse(cumsum(reverse(v,dims=1),dims=1),dims=1),
		ARDT,ARDT,name="cumsum operator")

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 ARDT,ARDT,name="Cumsum operator")
end

function DiffOp(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nt = J.recGeometry.nt[1]

	C = joLinearFunctionFwd_T(nt, nt,
		v -> [diff(v,dims=1);zeros(ARDT,1,size(v,2))],
		w -> reverse([diff(cumsum(reverse(w,dims=1),dims=1));zeros(ARDT,1,size(w,2))],dims=1),
		ARDT,ARDT,name="diff operator")

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 ARDT,ARDT,name="Diff operator")
end

function DiffOp(F::judiPDEextended{ADDT,ARDT}) where {ADDT,ARDT}
	nt = F.recGeometry.nt[1]

	C = joLinearFunctionFwd_T(nt, nt,
		v -> [diff(v,dims=1);zeros(ARDT,1,size(v,2))],
		w -> reverse([diff(cumsum(reverse(w,dims=1),dims=1));zeros(ARDT,1,size(w,2))],dims=1),
		ARDT,ARDT,name="diff operator")

	P = joLinearFunctionFwd_T(size(F,1), size(F,1),
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 ARDT,ARDT,name="Diff operator")
end

function HammingOp(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
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

function FractionalIntegrationOp(J::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
	nsrc = length(J.srcGeometry.xloc)
	nt = J.recGeometry.nt[1]

	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{ARDT,2},fgl_deriv(-0.5,x,1));	# half integration
	C    = joConvolve(nt,1,y[:];DDT=ARDT,RDT=ARDT);

	P = joLinearFunctionFwd_T(size(J,1), size(J,1),
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 ARDT,ARDT,name="Fractional integration operator")
end

function FractionalIntegrationOp(F::judiPDEextended{ADDT,ARDT}) where {ADDT,ARDT}
	nt = F.recGeometry.nt[1]

	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{ARDT,2},fgl_deriv(0.95,x,1));	# half integration
	C    = joConvolve(nt,1,y[:];DDT=ARDT,RDT=ARDT);

	P = joLinearFunctionFwd_T(size(F,1), size(F,1),
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 ARDT,ARDT,name="Fractional integration operator")
end

function FractionalIntegrationOp(nt::Int,nsrc::Int,nrec::Int,order::Number;DDT=Float32)

	x    = zeros(Int64,nt,1);
	x[1] = 1;
	y    = convert(Array{DDT,2},fgl_deriv(order,x,1));	# fractional integration
	C    = joConvolve(nt,1,y[:];DDT=DDT,RDT=DDT);

	P = joLinearFunctionFwd_T(nt*nsrc*nrec, nt*nsrc*nrec,
	                             v -> apply_frac_integral(C,v),
	                             w -> apply_frac_integral(C',w),
								 DDT,DDT,name="Fractional integration operator")
end

# right preconditioners: work in model domain

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