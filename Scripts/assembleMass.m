function A=assembleMass(nvx,nvy,hx,hy)

Mref=[1/3 1/6; 1/6 1/3];

Mx=hx*Mref;
My=hy*Mref;

Aloc=kron(My,Mx);

nv=nvx*nvy;
ne=(nvx-1)*(nvy-1);

id=1:nv;
id=reshape(id,nvx,nvy);

a=id(1:end-1,1:end-1); a=a(:)';
b=id(2:end,1:end-1);  b=b(:)';
c=id(1:end-1,2:end); c=c(:)';
d=id(2:end,2:end); d=d(:)';

conn=[a;b;c;d];

ii=(1:4)';
ii=repmat(ii,[1 4]);
jj=ii';

I=conn(ii(:),:);
J=conn(jj(:),:);

V=repmat(Aloc(:),[1 ne]);

A=sparse(I,J,V);
