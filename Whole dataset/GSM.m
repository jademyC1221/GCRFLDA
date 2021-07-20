function [ miRNA_gsSim ] = GSM( md_adjMat )
%GSM Summary of this function goes here
%   Detailed explanation goes here

nm=size(md_adjMat,1);
normSum=0;
for i=1:nm
    
   normSum=normSum+ ((norm(md_adjMat(i,:),2)).^2);
    
end

rm=1/(normSum/nm);

miRNA_gsSim = zeros(nm,nm);

for i=1:nm
   for j=1:nm
       sub=md_adjMat(i,:)-md_adjMat(j,:);
       miRNA_gsSim(i,j)=exp(-rm*((norm(sub,2)).^2));
       
   end 
    
end

end

