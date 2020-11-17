using PyPlot, JUDI.TimeModeling, LinearAlgebra, FFTW, JOLI, DSP
using IterativeSolvers

using SeismicPreconditioners

n = (401, 401)   # (x,y,z) or (x,z)
d = (2.5f0, 2.5f0)
o = (0f0, 0f0)

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

v0 = 2f0*ones(Float32,n)
v = deepcopy(v0)
v[201,201] = 3f0  # dirac perturbation

# Slowness squared [s^2/km^2]
m = convert(Array{Float32,2},(1f0 ./ v).^2)	# true model
m0 = convert(Array{Float32,2},(1f0 ./ v0).^2)
dm = vec(m-m0)

model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Model structure

dtS = 1f0
timeS = 1500f0
nt = Int64(timeS/dtS)+1

fmin = 10f0
fmax = 80f0
cfreqs = (fmin, fmin+5f0, fmax-5f0, fmax) # corner frequencies
wavelet = cfreq_wavelet(500, nt, dtS/1f3, cfreqs; edge=hamming)
#wavelet = ricker_wavelet(timeS, dtS, 0.03f0)

################################### Source/receiver geometry ######################################

# Set up receiver geometry

nrec = n[1]

xrec = range(0f0, stop=Float32((n[1]-1)*d[1]), length=nrec)
yrec = 0f0
zrec = range(14f0, stop=14f0, length=nrec)

# receiver sampling and recording time
timeR = timeS   # receiver recording time [ms]
dtR   = dtS 
# Set up receiver structure
nsrc = 16

xsrc = convertToCell(range(0f0, stop=Float32((n[1]-1)*d[1]), length=nsrc))
# xsrc = convertToCell(range((n[1]-1)*d[1]/2f0, stop=(n[1]-1)*d[1]/2f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(9f0, stop=9f0, length=nsrc))


recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

srcGeometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dtS, t=timeS)
#srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model0)
info = Info(prod(n), nsrc, ntComp)

opt = Options(free_surface=false,isic=false)

Pr  = judiProjection(info, recGeometry)
F0  = judiModeling(info, model0; options=opt) # modeling operator background model
Ps = judiProjection(info, srcGeometry)
# Combined operators
F0 = Pr*F0*adjoint(Ps)

J = judiJacobian(F0, q)
P = RightPrecondJ(J)

PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure(); imshow(reshape(sqrt.(1f0./m), n[1], n[2])',extent=(0,extentx,extentz,0));
PyPlot.scatter(xrec,zrec,marker=".",color="red",label="receivers",s=1);
PyPlot.scatter(xsrc,zsrc,marker=".",color="white",label="sources",s=1);
legend(loc=3,fontsize=8);
xlabel("Lateral position [m]", fontsize=8);ylabel("Horizontal position [m]", fontsize=8);
fig = PyPlot.gcf();
title("Experimental setup")

######################################### Set up JUDI operators ###################################

d_lin = J*dm

dm1 = J'*d_lin

dm2 = P'*J'*J*P*dm

figure();
subplot(2,1,1);
PyPlot.plot(wavelet);title("zoomed-in wavelet in time domain");
subplot(2,1,2);
PyPlot.plot(abs.(fftshift(fft(wavelet))));title("wavelet in frequency domain");

figure();
subplot(2,2,1)
imshow(dm1.data');title("J'*J*dm")
subplot(2,2,2)
imshow(reshape(dm2,n)');title("P'*J'*J*P*dm")
subplot(2,2,3)
imshow(abs.(fftshift(fft(dm1.data')))/norm(abs.(fftshift(fft(dm1.data'))),Inf),cmap="jet",vmin=0,vmax=1)
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(dm2,n)')))/norm(abs.(fftshift(fft(reshape(dm2,n)'))),Inf),cmap="jet",vmin=0,vmax=1)

print(stop)

L = laplacian_operator(model0,-1,J.info)

figure();
subplot(1,2,1)
imshow(reshape(dm3,n)');title("P'*J'*J*P*dm")
subplot(1,2,2)
imshow(abs.(fftshift(fft(reshape(dm3,n)')))/norm(abs.(fftshift(fft(reshape(dm3,n)'))),Inf),cmap="jet",vmin=0,vmax=1)


dm3 = L*dm1

temp = deepcopy(abs.(fftshift(fft(dm1.data))))
temp[95:105,95:105] .= 0 
figure();imshow(temp')

figure();
subplot(2,2,1)
imshow(dm1.data');title("J'*J*dm")
subplot(2,2,2)
imshow(reshape(dm2,n)');title("P'*J'*J*P*dm")
subplot(2,2,3)
imshow(abs.(fftshift(fft(dm1.data')))/norm(abs.(fftshift(fft(dm1.data'))),Inf),cmap="jet",vmin=0,vmax=0.5)
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(dm2,n)')))/norm(abs.(fftshift(fft(reshape(dm2,n)'))),Inf),cmap="jet",vmin=0,vmax=0.5)


maxit = 20

x1 = 0f0 .* dm
x1,his1 = lsqr!(x1,J,d_lin,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
x2 = 0f0 .* dm
x2,his2 = lsqr!(x2,J*P,d_lin,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
res1 = his1[:resnorm]
res2 = his2[:resnorm]

u1 = 20*log.(res1./norm(d_lin))
u2 = 20*log.(res2./norm(d_lin))

figure();
xlabel("iterations")
ylabel("ratio")
PyPlot.plot(res1/norm(d_lin),label="LSQR")
PyPlot.plot(res2/norm(d_lin),label="P-LSQR")
title("normalized least squares residual")
legend()


figure();
xlabel("iterations")
ylabel("dB")
PyPlot.plot(u1,label="LSQR")
PyPlot.plot(u2,label="P-LSQR")
title("normalized log-based least squares residual")
legend()

figure();
subplot(2,2,1)
imshow(reshape(x1,n)');title("LSQR")
subplot(2,2,2)
imshow(reshape(P*x2,n)');title("P-LSQR")
subplot(2,2,3)
imshow(abs.(fftshift(fft(reshape(x1,n)')))/norm(abs.(fftshift(fft(reshape(x1,n)'))),Inf),cmap="jet",vmin=0,vmax=1)
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(P*x2,n)')))/norm(abs.(fftshift(fft(reshape(P*x2,n)'))),Inf),cmap="jet",vmin=0,vmax=1)