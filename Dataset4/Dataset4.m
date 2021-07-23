[~,lncname]=xlsread('associationLDNFSGB.xlsx','Sheet2');  
[~,disname]=xlsread('associationLDNFSGB.xlsx','Sheet3'); 
[~,association]=xlsread('associationLDNFSGB.xlsx','Sheet1');  
load LDNFSGB_lncdiamatrix;

% 
% M = [];
% for i=1:1765
%     n = find(strcmp(lncname, lncrnaDisease(i,1)));
%     m = find(strcmp(disname, lncrnaDisease(i,2)));
%     N = [n m];
%     M = [M;N];
% end
% 
% nl=max(M(:,1));
% nm=max(M(:,2));
% pp=size(M,1);
% interaction=zeros(nl,nm);
% for i=1:pp
%     interaction(M(i,1),M(i,2))=1;
% end
% LDNFSGB_lncdiamatrix = interaction;
% save LDNFSGB_lncdiamatrix LDNFSGB_lncdiamatrix;

lncrnaDisease = LDNFSGB_lncdiamatrix;
% r lncid, c disid
[r,c] = find(lncrnaDisease);
%540*2
gld = [r c];  
pp = size(gld,1);
[r0,c0] = find(lncrnaDisease==0);
%19930*2
fgld=[r0 c0];  
fpp = size(fgld,1);
x1=randperm(pp)';
positive=gld(x1,:);
x2=randperm(fpp)';
negative=fgld(x2,:);

% use 1765 positives and 1765 negtives--5fold cv
for cv=1:10
    cv
    fprintf('cv=%d\n',cv)
    
    x1=randperm(pp)';
    positive=gld(x1,:);

    for ccv=1:5
        ccv
        %floor(pp/5)=353
        if ccv==1
            postest = positive(x1((ccv-1)*floor(pp/5)+1:floor(pp/5)*ccv),:);
            numpostest = length(postest);
            x2 = randperm(fpp)';
            negative = fgld(x2(1:numpostest),:);
            test = [postest;negative];
            labeltest = [ones(floor(pp/5),1);zeros(numpostest,1)];
            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(floor(pp/5)+numpostest,1)*1111;
            edge_f1test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(floor(pp/5)+numpostest)';
            edge_f11 = edge_f1test(x3,:);

            postrain = positive(x1(floor(pp/5)*ccv+1:pp),:); 
            numneg = length(postrain);
            numtrain = numneg*2;
            negtrain = fgld(x2(1:numneg),:);
            train = [postrain;negtrain];
            labeltrain = [ones(numneg,1);zeros(numneg,1)];
            lnc_id = train(:,1);
            dis_id = train(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usagetr = ones(numtrain,1)*2222;
            edge_f1train = [train labeltrain lnc_id dis_id usagetr];
            x3 = randperm(numtrain)';
            edge_f12 = edge_f1train(x3,:);
            
            edge_f1 = [edge_f12;edge_f11];
            xlswrite(['edge_f_ldnfsgb_cv3530_',num2str(cv),'.xlsx'],edge_f1,'Sheet 1')

        elseif ccv<5
            postest = positive(x1((ccv-1)*floor(pp/5)+1:floor(pp/5)*ccv),:);
            numpostest = length(postest);
            x2=randperm(fpp)';
            negative=fgld(x2(1:numpostest),:);
            test = [postest;negative];
            labeltest = [ones(numpostest,1);zeros(numpostest,1)];

            postrain1 = positive(x1(1:(ccv-1)*floor(pp/5)),:);
            postrain2 = positive(x1(ccv*floor(pp/5)+1:pp),:);
            postrain = [postrain1;postrain2];
            numneg = length(postrain);
            numtrain = numneg*2;
            negtrain = fgld(x2(1:numneg),:);
            train = [postrain;negtrain];
            labeltrain = [ones(numneg,1);zeros(numneg,1)];

            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numpostest*2,1)*1111;    
            
            if ccv==2
                edge_f2test = [test labeltest lnc_id dis_id usaget];
                x3 = randperm(numpostest*2)';
                edge_f21 = edge_f2test(x3,:);
                lnc_id = train(:,1);
                dis_id = train(:,2);
                lnc_id = lnc_id-1;
                dis_id = dis_id-1;
                usagetr = ones(numtrain,1)*2222;
                edge_f2train = [train labeltrain lnc_id dis_id usagetr];
                x3 = randperm(numtrain)';
                edge_f22 = edge_f2train(x3,:);
                edge_f2 = [edge_f22;edge_f21];
                xlswrite(['edge_f_ldnfsgb_cv3530_',num2str(cv),'.xlsx'],edge_f2,'Sheet 2')
            elseif ccv==3
                edge_f3test = [test labeltest lnc_id dis_id usaget];
                x3 = randperm(numpostest*2)';
                edge_f31 = edge_f3test(x3,:);
                lnc_id = train(:,1);
                dis_id = train(:,2);
                lnc_id = lnc_id-1;
                dis_id = dis_id-1;
                usagetr = ones(numtrain,1)*2222;
                edge_f3train = [train labeltrain lnc_id dis_id usagetr];
                x3 = randperm(numtrain)';
                edge_f32 = edge_f3train(x3,:);
                edge_f3 = [edge_f32;edge_f31];
                xlswrite(['edge_f_ldnfsgb_cv3530_',num2str(cv),'.xlsx'],edge_f3,'Sheet 3')
            else ccv==4
                edge_f4test = [test labeltest lnc_id dis_id usaget];
                x3 = randperm(numpostest*2)';
                edge_f41 = edge_f4test(x3,:);
                lnc_id = train(:,1);
                dis_id = train(:,2);
                lnc_id = lnc_id-1;
                dis_id = dis_id-1;
                usagetr = ones(numtrain,1)*2222;
                edge_f4train = [train labeltrain lnc_id dis_id usagetr];
                x3 = randperm(numtrain)';
                edge_f42 = edge_f4train(x3,:);
                edge_f4 = [edge_f42;edge_f41];
                xlswrite(['edge_f_ldnfsgb_cv3530_',num2str(cv),'.xlsx'],edge_f4,'Sheet 4')
            end

        else
            postest = positive(x1((ccv-1)*floor(pp/5)+1:pp),:);
            numpostest = length(postest);
            x2 = randperm(fpp)';
            negative = fgld(x2(1:numpostest),:);
            test = [postest;negative];
            labeltest = [ones(numpostest,1);zeros(numpostest,1)];  %539个test样本
            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numpostest*2,1)*1111;
            edge_f5test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(numpostest*2)';
            edge_f51 = edge_f5test(x3,:);
            
            numpostrain = pp-numpostest;
            postrain = positive(x1(1:numpostrain),:);
            negtrain = fgld(x2(1:numpostrain),:);
            train = [postrain;negtrain];
            labeltrain = [ones(numpostrain,1);zeros(numpostrain,1)];
            lnc_id = train(:,1);
            dis_id = train(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usagetr = ones(numpostrain*2,1)*2222;
            edge_f5train = [train labeltrain lnc_id dis_id usagetr];
            x3 = randperm(numpostrain*2)';
            edge_f52 = edge_f5train(x3,:);
            edge_f5 = [edge_f52;edge_f51];
            xlswrite(['edge_f_ldnfsgb_cv3530_',num2str(cv),'.xlsx'],edge_f5,'Sheet 5')
        end

    end
    
end

% Five fold cross-validation 所有负样本
for cv=1:5
    cv
    fprintf('cv=%d\n',cv)
    
    x1=randperm(pp)';
    positive=gld(x1,:);

    for ccv=1:5
        ccv
        %floor(pp/10)
        if ccv==1
            postest = positive(x1((ccv-1)*floor(pp/5)+1:floor(pp/5)*ccv),:);
            numpostest = length(postest);
            x2 = randperm(fpp)';
            negative = fgld(x2,:);
            test = [postest;negative];
            numtest = length(test);
            labeltest = [ones(numpostest,1);zeros(fpp,1)];
            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numtest,1)*1111;
            edge_f1test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(numtest)';
            edge_f11 = edge_f1test(x3,:);

            postrain = positive(x1(floor(pp/5)*ccv+1:pp),:); 
            numpostrain = length(postrain);
            negtrain = fgld(x2(1:numpostrain),:);                      
            train = [postrain;negtrain];
            numtrain = length(train);
            labeltrain = [ones(numpostrain,1);zeros(numpostrain,1)];
            lnc_id = train(:,1);
            dis_id = train(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usagetr = ones(numtrain,1)*2222;
            edge_f1train = [train labeltrain lnc_id dis_id usagetr];
            x3 = randperm(numtrain)';
            edge_f12 = edge_f1train(x3,:);
            
            edge_f1 = [edge_f12;edge_f11];
            xlswrite(['LDNFSGB_5cvallneg_',num2str(cv),'.xlsx'],edge_f1,'Sheet 1')

        elseif ccv<5
            postest = positive(x1((ccv-1)*floor(pp/5)+1:floor(pp/5)*ccv),:);
            numpostest = length(postest);
            x2=randperm(fpp)';
            negative=fgld(x2,:);
            test = [postest;negative];
            numtest = length(test);
            labeltest = [ones(numpostest,1);zeros(fpp,1)];

            postrain1 = positive(x1(1:(ccv-1)*floor(pp/5)),:);
            postrain2 = positive(x1(ccv*floor(pp/5)+1:pp),:);
            postrain = [postrain1;postrain2];
            numpostrain = length(postrain);
            negtrain = fgld(x2(1:numpostrain),:);
            train = [postrain;negtrain];
            numtrain = length(train);
            labeltrain = [ones(numpostrain,1);zeros(numpostrain,1)];

            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numtest,1)*1111;    
            
            edge_f2test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(numtest)';
            edge_f21 = edge_f2test(x3,:);
            lnc_id = train(:,1);
            dis_id = train(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usagetr = ones(numtrain,1)*2222;
            edge_f2train = [train labeltrain lnc_id dis_id usagetr];
            x3 = randperm(numtrain)';
            edge_f22 = edge_f2train(x3,:);
            edge_f2 = [edge_f22;edge_f21];
            xlswrite(['LDNFSGB_5cvallneg_',num2str(cv),'.xlsx'],edge_f2,['Sheet ',num2str(ccv)]);

        else
            postest = positive(x1((ccv-1)*floor(pp/5)+1:pp),:);
            x2 = randperm(fpp)';
            negative = fgld(x2,:);
            test = [postest;negative];
            numtest = length(test);
            labeltest = [ones(pp-floor(pp/5)*4,1);zeros(fpp,1)];
            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numtest,1)*1111;
            edge_f10test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(numtest)';
            edge_f51 = edge_f10test(x3,:);

            postrain = positive(x1(1:(ccv-1)*floor(pp/5)),:);
            numpostrain = length(postrain);
            negtrain = fgld(x2(1:(ccv-1)*floor(pp/5)),:);
            train = [postrain;negtrain];
            numtrain = length(train);
            labeltrain = [ones(numpostrain,1);zeros(numpostrain,1)];
            lnc_id = train(:,1);
            dis_id = train(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usagetr = ones(numtrain,1)*2222;
            edge_f10train = [train labeltrain lnc_id dis_id usagetr];
            x3 = randperm(numtrain)';
            edge_f52 = edge_f10train(x3,:);
            edge_f5 = [edge_f52;edge_f51];
            xlswrite(['LDNFSGB_5cvallneg_',num2str(cv),'.xlsx'],edge_f5,['Sheet ',num2str(ccv)])
        end

    end
    
end

%%
[ disease_gsSim ] = GSD( lncrnaDisease );
[ lncRNA_gsSim ] = GSM( lncrnaDisease );
[ Coslnc,CosDis] = cosSim( lncrnaDisease );

%
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
%%
sim1=lncRNA_gsSim;
sim2=Coslnc;
for i=1:9
    w=F2(i,:);
    [ sim ] = LKFtwo( w,sim1,sim2 );
    
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    simname = strcat('ldnfsgb_lncsim',h);

    eval([simname '=sim']);
    save([simname],[simname])  %保存变化的变量名
    
end

sim1=disease_gsSim;
sim2=CosDis;
for i=1:9
    w=F2(i,:);
    [ sim ] = LKFtwo( w,sim1,sim2 );
    
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    simname = strcat('ldnfsgb_dissim',h);
    eval([simname '=sim']);
    save([simname],[simname])  %保存变化的变量名
    
end

% E2=[];
% for i=1:9
%     h=Fname2(i,:);
%     h1=num2str(h(1));h2=num2str(h(2));
%     h=strcat(h1,h2);
%     E2=[E2;h];
% end

%%  case study Dataset4
clear;
load 'D:\Study研究生\GCRFLDA\Myproject\对比试验\LDNFSGB_lncdiamatrix.mat';

lncrnaDisease = LDNFSGB_lncdiamatrix;
% known association
[r,c] = find(lncrnaDisease);
gld=[r c];     %2697*2
pp=length(gld);
% unknown association
[r0,c0] = find(lncrnaDisease==0);
fgld=[r0 c0];  %96183*2
fpp=length(fgld);
%train
x1=randperm(pp)';
postrain=gld(x1,:);
x2=randperm(fpp)';  
negtrain = fgld(x2(1:pp),:);                      
train = [postrain;negtrain];
labeltrain = [ones(pp,1);zeros(pp,1)];

lnc_id = train(:,1);
dis_id = train(:,2);
lnc_id = lnc_id-1;
dis_id = dis_id-1;
usagetr = ones(pp*2,1)*2222;
edge_f1train = [train labeltrain lnc_id dis_id usagetr];

x3 = randperm(pp*2)';
edge_f12 = edge_f1train(x3,:);

%test
x2 = randperm(fpp)';
negative = fgld(x2,:);
test = negative;   
labeltest = [zeros(fpp,1)];
lnc_id = test(:,1);
dis_id = test(:,2);
lnc_id = lnc_id-1;
dis_id = dis_id-1;
usaget = ones(fpp,1)*1111;
edge_f1test = [test labeltest lnc_id dis_id usaget];
edge_f11 = edge_f1test;

edge_f1 = [edge_f12;edge_f11];

xlswrite(['Dataset4casestudy.xlsx'],edge_f1,'Sheet 1');
fprintf('dataset4 finish!\n');

%%
clear;
predY =xlsread('predYDataset4.xlsx');
[~,lncname]=xlsread('associationLDNFSGB.xlsx','Sheet2');  
[~,disname]=xlsread('associationLDNFSGB.xlsx','Sheet3');  
length(predY);
Disname = [];
for i=1:length(predY)
    
    n = disname(predY(i,2));
    Disname = [Disname;n];
end

Lncname = [];
for i=1:length(predY)
    
    n = lncname(predY(i,1));
    Lncname = [Lncname;n];
end

[~,lncrna] = xlsread('./dataset4.xlsx','Sheet1');
[~,ALLDatabase] = xlsread('./ALLDATABASES3.xlsx','Sheet1');
PMID = ALLDatabase(:,4);
ALLlnc = ALLDatabase(:,2);
ALLdis = ALLDatabase(:,3);
ALLlncdis = ALLDatabase(:,1);

Dataset4gcrflda_case=[]
for i=1:1500
    i
    for j=1:length(ALLlncdis)
        if strcmp(lncrna(i),ALLlncdis(j))
            OUT = [i ALLlnc(j) ALLdis(j) PMID(j)];
            Dataset4gcrflda_case=[Dataset4gcrflda_case;OUT];
        end
    end
end

% save Dataset4gcrflda_case Dataset4gcrflda_case;