%
[~,lncname]=xlsread('association-train2697.xlsx','Sheet2');  % 972RNA 646dis
[~,disname]=xlsread('association-train2697.xlsx','Sheet3');  % 972RNA 646dis
[~,lncrnaDisease]=xlsread('association-train2697.xlsx','Sheet1');  % 4339 association

% M = [];
% for i=1:2697
%     n = find(strcmp(lncname, lncrnaDisease(i,1)));
%     m = find(strcmp(disname, lncrnaDisease(i,2)));
%     N = [n m];
%     M = [M;N];
% end
% 
% nl = 240;
% nm = 412;
% pp=size(M,1);
% interaction=zeros(nl,nm);
% for i=1:pp
%     interaction(M(i,1),M(i,2))=1;
% end
% interaction=lncrnaDisease;

load lncrnaDisease.mat;

% known association
[r,c] = find(lncrnaDisease);
gld=[r c];     %2697*2
pp=length(gld);
% unknown association
[r0,c0] = find(lncrnaDisease==0);
fgld=[r0 c0];  %96183*2
fpp=length(fgld);

x1=randperm(pp)';
positive=gld(x1,:);
x2=randperm(fpp)';
negative=fgld(x2,:);

% Five fold cross-validation
for cv=1:5
    cv
    fprintf('cv=%d\n',cv)
    
    x1=randperm(pp)';
    positive=gld(x1,:);

    for ccv=1:5
        ccv
        %floor(pp/5)=
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
            xlswrite(['edge_f5cv',num2str(cv),'.xlsx'],edge_f1,'Sheet 1')

        elseif ccv<5
            postest = positive(x1((ccv-1)*floor(pp/5)+1:floor(pp/5)*ccv),:);
            numpostest = length(postest);
            x2=randperm(96183)';
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
            xlswrite(['edge_f5cv',num2str(cv),'.xlsx'],edge_f2,['Sheet ',num2str(ccv)]);

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
            xlswrite(['edge_f5cv',num2str(cv),'.xlsx'],edge_f5,['Sheet ',num2str(ccv)])
        end

    end
    
end

%%
E2=[];
for i=1:9
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    E2=[E2;h];
end
save E2 E2;

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
%LNF for lncRNA similarity
sim1=lncRNA_gsSim;
sim2=Coslnc;
for i=1:9
    w=F2(i,:);
    [ sim ] = LKFtwo( w,sim1,sim2 );
    
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    simname = strcat('lncsim',h);

    eval([simname '=sim']);
    save([simname],[simname])  %保存变化的变量名
    
end
%LNF for disease similarity
sim1=disease_gsSim;
sim2=CosDis;
for i=1:9
    w=F2(i,:);
    [ sim ] = LKFtwo( w,sim1,sim2 );
    
    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    simname = strcat('dissim',h);
    eval([simname '=sim']);
    save([simname],[simname])  %保存变化的变量名
    
end

%% case study Dataset1
load lncrnaDisease.mat;
dataset = lncrnaDisease;

lncrnaDisease = dataset;
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

xlswrite(['Dataset1casestudy.xlsx'],edge_f1,'Sheet 1');
fprintf('dataset1 finish!\n');

%Dataset 1
predY =xlsread('predYDataset1.xlsx','Sheet 1');
[~,lncname]=xlsread('association-train2697.xlsx','Sheet2');  
[~,disname]=xlsread('association-train2697.xlsx','Sheet3');  
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

%search from databases
[~,lncrna] = xlsread('predYDataset1.xlsx','Sheet3');
[~,ALLDatabase] = xlsread('./ALLDATABASES3.xlsx','Sheet1');
PMID = ALLDatabase(:,4);
ALLlnc = ALLDatabase(:,2);
ALLdis = ALLDatabase(:,3);
ALLlncdis = ALLDatabase(:,1);

Dataset1gcrflda_case=[]
for i=1:1500
    i
    for j=1:37888
        if strcmp(lncrna(i),ALLlncdis(j))
            OUT = [i ALLlnc(j) ALLdis(j) PMID(j)];
            Dataset1gcrflda_case=[Dataset1gcrflda_case;OUT];
        end
    end
end

save Dataset1gcrflda_case Dataset1gcrflda_case;