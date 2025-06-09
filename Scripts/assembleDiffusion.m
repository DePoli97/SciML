function A=assembleDiffusion(nvx,nvy,hx,hy)
%%
%% nvx number of vertices along x
%% nvy number of vertices along y
%% mesh size along x
%% mesh size along y

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

% local stiffness matrix
Aloc=kron(My,Ax)+kron(Ay,Mx);

% create connectivity matrix
nv=nvx*nvy;
ne=(nvx-1)*(nvy-1);

id=1:nv;
id=reshape(id,nvx,nvy);

a=id(1:end-1,1:end-1); a=a(:)';
b=id(2:end,1:end-1);  b=b(:)';
c=id(1:end-1,2:end); c=c(:)';
d=id(2:end,2:end); d=d(:)';

conn=[a;b;c;d];
% done

% create data structure to assemble
% using sparse
ii=(1:4)';
ii=repmat(ii,[1 4]);
jj=ii';

I=conn(ii(:),:);
J=conn(jj(:),:);

% matrix containing per column
% the local stiffness matrices
V=repmat(Aloc(:),[1 ne]);

A=sparse(I,J,V);
