export fgl_deriv

function fgl_deriv( a, y, h )
    #   Computes the fractional derivative of order alpha (a) for the function
    #   y sampled on a regular grid with spacing h, using the Grunwald-Letnikov
    #   formulation.
    #   Note that this implementation is similar to that of Bayat 2007
    #   (FileExchange: 13858-fractional-differentiator), and takes the exact
    #   same inputs, but uses a vectorized formulation to accelerates the
    #   computation in Matlab.
    #   Copyright 2014 Jonathan Hadida
    #   Contact: Jonathan dot hadida [a] dtc.ox.ac.uk
        #using LinearAlgebra SpecialFunctions # added for Julia 1.1
        n     = length(y);
        J     = collect(0:(n-1));
        #G1    = gamma.( J+1 );
        #G2    = gamma.( a+1-J );
        G1    = gamma.( J+convert(Array{Int64,1},ones(size(J))));
        test21 = convert(Array{Float64,1},J);
        test22 = (a+1)*ones(size(J));
        G2    = gamma.(-test21+test22);
        s     = (-1) .^ J;
    
        #M     = tril( ones(n,n) );
        M     = LowerTriangular( ones(n,n) );
        test  = y;
        R     = zeros(n,n);
        for i = 1:n
            R[i,i:n] = y[1:n - i + 1]
            R[i:n,i] = y[1:n - i + 1]
        end
        #T     = repmat( (gamma.(a+1)/(h^a)) * s ./ (G1.*G2),1,size(M,2));
        T     = repeat( (gamma.(a+1)/(h^a)) * s ./ (G1.*G2),1,size(M,2));
        #Y     = reshape(sum( R .* M .* T, 2 ), size(y));
        Y     = reshape(sum( R .* M .* T, dims=2 ), size(y));
        #Y[isnan.(Y)] = 0;
        inds = findall(isnan,Y);
        for i = 1:length(inds)
        Y[inds[i]] = 0
        end
        return Y
    end
    