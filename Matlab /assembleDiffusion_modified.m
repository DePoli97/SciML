function A=assembleDiffusion(nvx,nvy,hx,hy,sigma_elements)
%%
%% Modified assembleDiffusion function to accept element-wise diffusivity
%% nvx: number of vertices along x
%% nvy: number of vertices along y
%% hx: mesh size along x
%% hy: mesh size along y
%% sigma_elements: vector of diffusivity values for each element (size: ne x 1)
%%

% reference diffusion matrix
Aref=[1 -1; -1 1];
% reference mass matrix
Mref=[1/3 1/6; 1/6 1/3];

% 1D diffusion matrix along x
Ax=1/hx*Aref;
% 1D diffusion matrix along y
Ay=1/hy*Aref;

% 1D mass matrix along x
Mx=hx*Mref;
% 1D mass matrix along y
My=hy*Mref;

% create connectivity matrix
nv=nvx*nvy;
ne=(nvx-1)*(nvy-1);

% Check if sigma_elements has correct size
if length(sigma_elements) ~= ne
    error('sigma_elements must have length equal to number of elements (%d)', ne);
end

id=1:nv;
id=reshape(id,nvx,nvy);

a=id(1:end-1,1:end-1); a=a(:)';
b=id(2:end,1:end-1);  b=b(:)';
c=id(1:end-1,2:end); c=c(:)';
d=id(2:end,2:end); d=d(:)';

conn=[a;b;c;d];

% create data structure to assemble using sparse
ii=(1:4)';
ii=repmat(ii,[1 4]);
jj=ii';

I=conn(ii(:),:);
J=conn(jj(:),:);

% Create local stiffness matrices for each element with different diffusivity
V = zeros(16, ne);
for e = 1:ne
    % local stiffness matrix for element e with diffusivity sigma_elements(e)
    Aloc = sigma_elements(e) * (kron(My,Ax) + kron(Ay,Mx));
    V(:,e) = Aloc(:);
end

A=sparse(I,J,V);