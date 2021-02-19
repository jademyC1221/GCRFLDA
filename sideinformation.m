
function [ lnRNA_gsSim ] = GSM( md_adjMat )
%  Gaussian Interaction Profile Kernel Similarity

    nm=size(md_adjMat,1);
    normSum=0;
    for i=1:nm

       normSum=normSum+ ((norm(md_adjMat(i,:),2)).^2);

    end

    rm=1/(normSum/nm);

    lnRNA_gsSim = zeros(nm,nm);

    for i=1:nm
       for j=1:nm
           sub=md_adjMat(i,:)-md_adjMat(j,:);
           lnRNA_gsSim(i,j)=exp(-rm*((norm(sub,2)).^2));

       end 

    end

end

function [ disease_gsSim ] = GSD( ld_adjMat )
%  Gaussian Interaction Profile Kernel Similarity
    
    nd=size(ld_adjMat,2);
    normSum=0;
    for i=1:nd

       normSum=normSum+ ((norm(ld_adjMat(:,i),2)).^2);

    end

    rd=1/(normSum/nd);

    disease_gsSim=zeros(nd,nd);

    for i=1:nd
       for j=1:nd
           sub=ld_adjMat(:,i)-ld_adjMat(:,j);
           disease_gsSim(i,j)=exp(-rd*((norm(sub,2)).^2));

       end 

    end

end

load lncrnaDisease.mat;
[ disease_gsSim ] = GSD( lncrnaDisease );
[ lncRNA_gsSim ] = GSM( lncrnaDisease );
save lncRNA_gsSim lncRNA_gsSim;
save disease_gsSim  disease_gsSim;


function [ CosLn,CosDi] = cosSim( data )
%  Cosine Similarity for Diseases
  
    rows=size(data,1);
    CosLn=zeros(rows,rows);

    for i=1:rows
        
        for j=1:i
            
            if (norm(data(i,:))*norm(data(j,:))==0)
                
                CosLn(i,j)=0;
                
            else
                CosLn(i,j)=dot(data(i,:),data(j,:))/(norm(data(i,:))*norm(data(j,:)));
                
            end
            
            CosLn(j,i)=CosLn(i,j);
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

load lncrnaDisease.mat;
[ Coslnc,CosDis] = cosSim( lncrnaDisease );
save Coslnc Coslnc;
save CosDis CosDis;


%% LKF 
function [ sim ] = LKFtwo( w,sim1,sim2 )
    sim = w(1)*sim1 + w(2)*sim2;
end

% LKFforlncRNA
function LKFforlncRNA()
    load lncrnaDisease.mat;
    Fname2=[];
    for x=1:1:9
        for y=1:1:9
                if x+y==10
                   E=[x y];
                   Fname2=[Fname2;E];
                end
        end
    end
    F2=Fname2/10;
    
    [ lncRNA_gsSim ] = GSM( lncrnaDisease );
    sim1=lncRNA_gsSim;
    [ Coslnc,CosDis] = cosSim( lncrnaDisease );
    sim2=Coslnc;
    
    
    for i=1:9
        w=F2(i,:);
        [ sim ] = LKFtwo( w,sim1,sim2 );

        h=Fname2(i,:);
        h1=num2str(h(1));h2=num2str(h(2));
        h=strcat(h1,h2);
        simname = strcat('lncsim',h);

        eval([simname '=sim']);
        save([simname],[simname])

    end

end


% LKFforDisease
function LKFforDisease()
    load lncrnaDisease.mat;
    Fname2=[];
    for x=1:1:9
        for y=1:1:9
                if x+y==10
                   E=[x y];
                   Fname2=[Fname2;E];
                end
        end
    end
    F2=Fname2/10;
    
    [ disease_gsSim ] = GSD( lncrnaDisease );
    sim1=disease_gsSim;
    [ Coslnc,CosDis] = cosSim( lncrnaDisease );
    sim2=CosDis;
    
    for i=1:9
        w=F2(i,:);
        [ sim ] = LKFtwo( w,sim1,sim2 );

        h=Fname2(i,:);
        h1=num2str(h(1));h2=num2str(h(2));
        h=strcat(h1,h2);
        simname = strcat('dissim',h);
        eval([simname '=sim']);
        save([simname],[simname])

    end
    
end


E2=[];
for i=1:9
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    E2=[E2;h];
end
save E2 E2;