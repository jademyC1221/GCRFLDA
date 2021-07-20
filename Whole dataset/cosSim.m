function [ CosMi,CosDi] = cosSim( data )
%COSSIM Summary of this function goes here
%   Detailed explanation goes here
   
rows=size(data,1);
CosMi=zeros(rows,rows);

    for i=1:rows
        
        for j=1:i
            
            if (norm(data(i,:))*norm(data(j,:))==0)
                
                CosMi(i,j)=0;
                
            else
                CosMi(i,j)=dot(data(i,:),data(j,:))/(norm(data(i,:))*norm(data(j,:)));
                
            end
            
            CosMi(j,i)=CosMi(i,j);
        end
        
        
    end
    
cols=size(data,2);
CosDi=zeros(cols,cols);

    for i=1:cols
        
        for j=1:i
            
            if (norm(data(:,i))*norm(data(:,j))==0)
                
                CosDi(i,j)=0;
                
            else
                CosDi(i,j)=dot(data(:,i),data(:,j))/(norm(data(:,i))*norm(data(:,j)));
                
            end
            
            CosDi(j,i)=CosDi(i,j);
        end
        
        
    end
    
end

