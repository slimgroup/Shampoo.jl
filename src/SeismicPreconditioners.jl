__precompile__()
module SeismicPreconditioners

using JUDI.TimeModeling, JOLI, LinearAlgebra, FFTW

include("RightPreconditioner/RightPreconditioner.jl")

include("utils/fractional_laplacian.jl")
include("utils/wavelet.jl")

end
