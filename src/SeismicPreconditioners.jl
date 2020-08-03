__precompile__()
module SeismicPreconditioners

using JUDI, JOLI, LinearAlgebra, FFTW

include("RightPreconditioner/RightPreconditioner.jl")

include("utils/fractional_laplacian.jl")
include("utils/wavelet.jl")

end
