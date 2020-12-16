export Filter_freq_band, createCirclesMask

function Filter_freq_band(Q,m,t, parm)
    n      = size(Q,1);
    Q      = [Q;zeros(Float32,size(Q))];
    d      = t[2]-t[1];
    dt     = collect(t[1]:d:(t[1]+2*n*d));
    Q_f    = zeros(Float32,size(Q));
    
    for i=1:size(Q,2);
	sf            = rfft(Q[:,i]);
        index         = length(sf);
        if m[1]*t[end]>0
          test        = collect(1:index);
          mask        = convert(Array{Float32,1},zeros(length(test)));
          #mask        = 1 ./ (1+exp(-parm*((test-round(m[1])))))+1 ./ (1+exp(+parm*((test-round(m[2])))));
	for j=1:length(test)
		mask[j] = 1/(1+exp(-parm*((test[j]-round(m[1])))))+1/(1+exp(+parm*((test[j]-round(m[2])))));
        end
        else
          test        = collect(1:index);
          #mask        = 1 ./ (1+exp(-parm*((test-round(m[2])))));
          mask        = convert(Array{Float32,1},zeros(length(test)));
	for j=1:length(test)
	  mask[j]     = 1/(1+exp(-parm*((test[j]-round(m[2])))));
        end	
        end
        #mask                          = mask -1;
	for j=1:length(mask)
          mask[j] = mask[j]-1;
        end
        sf	                      = mask .* sf;
	q_f		              = irfft(sf,size(Q,1));
	#q_f[convert(Int64,floor(length(q_f)/2)):end] = 0;
	a = convert(Int64,floor(length(q_f)/2));
	for j = a:length(q_f)
	  q_f[j] = 0;
	end
        Q_f[:,i]                      = q_f;
   end
   Q_f = Q_f[1:n,:];
   return Q_f
end


function createCirclesMask(image,centers,radii)
# Brett Shoelson, PhD
# 9/22/2014
# Comments, suggestions welcome: brett.shoelson@mathworks.com
# Copyright 2014 The MathWorks, Inc.
#IMAGE specified
xDim,yDim   = size(image);
xc          = centers[:,1];
yc          = centers[:,2];
x           = 1:xDim;
y           = 1:yDim;
#yy          = repmat(x[:], 1, length(y));
#xx          = repmat(y[:]', length(x), 1);
yy          = repeat(x[:], 1, length(y));
xx          = repeat(y[:]', length(x), 1);
mask        = falses(xDim,yDim);
mask1       = falses(xDim,yDim);
#for ii = 1:length(radii)
#	mask = mask | convert(Array{Int,2},floor(hypot.(xx - xc[ii], yy - yc[ii]))).<=radii[ii];
#end
#added for julia 1.1
for ii = 1:length(radii)
        testx = convert(Array{Float64,2},xx);
        testxc = convert(Array{Float64,2},xc[ii]*ones(size(testx)));
        testy = convert(Array{Float64,2},yy);
        testyc = convert(Array{Float64,2},yc[ii]*ones(size(testy)));
        test11=hypot.(testx - testxc, testy - testyc);
        test12 = zeros(size(test11));
        for i=1:size(test11,1)
       		for j=1:size(test11,2)
       		test12[i,j] = floor(test11[i,j])
       		end
        end
        test13=convert(Array{Int,2},test12).<=radii[ii]
        for i=1:size(mask1,1)
       		for j=1:size(mask1,2)
       		mask[i,j] = mask1[i,j] | test13[i,j];
       		end
        end
end
return mask
end


