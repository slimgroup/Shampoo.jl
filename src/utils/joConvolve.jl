# Extension operator: joConvolve
# Author : Rajiv Kumar
## helper module
module joConvolve_etc
    using JOLI: jo_convert 
    using FFTW
    ## forward 1D convolution
# function Convolve1D_fwd{VT<:Number}(x::AbstractVector{VT}, fKernel::AbstractVector,idx::AbstractVector, k::Integer,m::Integer,cflag::Bool,rdt=DataType)
function Convolve1D_fwd(x::AbstractVector, fKernel::AbstractVector,idx::AbstractVector, k::Integer,m::Integer,cflag::Bool,rdt=DataType)
        #fx = fft([full(x);zeros(VT,k-1,1)[:]]);
        #fx = fft([full(x);zeros(k-1,1)[:]]);
         fx = fft([x;zeros(k-1,1)[:]]);
         y = ifft(fKernel.*fx);
         y = y[idx];
         if ~cflag && isreal(y)
             y = real(y);
         end
        return jo_convert(rdt,y,false)
 end
  ## transpose 1D convolution
 #function Convolve1D_tran{VT<:Number}(x::AbstractVector{VT}, fKernel::AbstractVector,idx::AbstractVector, k::Integer,m::Integer,cflag::Bool,rdt=DataType)
 function Convolve1D_tran(x::AbstractVector, fKernel::AbstractVector,idx::AbstractVector, k::Integer,m::Integer,cflag::Bool,rdt=DataType)
        #z      = zeros(VT,m+k-1,1)[:];
        z      = zeros(m+k-1,1)[:];
        #z[idx] = full(x);
        z[idx] = x;
        y      = ifft(conj(fKernel).*fft(z));
        y      = y[1:m];
         if ~cflag && isreal(y)
             y = real(y);
         end
        return jo_convert(rdt,y,false)
 end
end
using .joConvolve_etc

export joConvolve


#function joConvolve{KDT<:Number}(m::Integer, n::Integer, kernel::AbstractVector{KDT}; DDT::DataType=joFloat,RDT::DataType=DDT)
function joConvolve(m::Integer, n::Integer, kernel::AbstractVector; DDT::DataType=joFloat,RDT::DataType=DDT)
                # convolution loop start
                offset = 1
                k      = length(kernel);
                cflag  = (eltype(kernel)<:Complex); #~isreal(kernel);
                # Shift kernel and add internal padding
                # kernel = [kernel[offset:end];zeros(KDT,m-1,1)[:];kernel[1:offset-1]];
                kernel = [kernel[offset:end];zeros(m-1,1)[:];kernel[1:offset-1]];
                   idx = collect(1:m);

                # Precompute kernel in frequency domain
               #fKernel = fft(full(kernel));
               fKernel = fft(kernel);
               # Convolution
               return joLinearFunctionFwd(length(idx),m,
                                        v1->joConvolve_etc.Convolve1D_fwd(v1,fKernel,idx, k,m,cflag,RDT),
                                        v2->joConvolve_etc.Convolve1D_tran(v2,fKernel,idx, k,m,cflag,DDT),
                                        v2->joConvolve_etc.Convolve1D_tran(v2,fKernel,idx, k,m,cflag,DDT),
                                        v1->joConvolve_etc.Convolve1D_fwd(v1,fKernel,idx, k,m,cflag,RDT),
                                        DDT,RDT;fMVok=false,
                                        name="joConvolve1D")
    return nothing
end

            

