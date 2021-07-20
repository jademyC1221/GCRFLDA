function [ disease_gsSim ] = GSD( md_adjMat )
%GSD Summary of this function goes here
%   Detailed explanation goes here
    
nd=size(md_adjMat,2);
normSum=0;
for i=1:nd
    
   normSum=normSum+ ((norm(md_adjMat(:,i),2)).^2);
    
end

rd=1/(normSum/nd);

disease_gsSim=zeros(nd,nd);

for i=1:nd
   for j=1:nd
       sub=md_adjMat(:,i)-md_adjMat(:,j);
       disease_gsSim(i,j)=exp(-rd*((norm(sub,2)).^2));
       
   end 
    
end

end

