% wholeset data
[~,lncrnaDisease]=xlsread('WholesetAssociation.xlsx','Sheet1');  % 4339 association
[~,lnDisname]=xlsread('WholesetAssociation.xlsx','Sheet2');  % 972RNA 646dis
lncname = lnDisname(:,1);
disname = lnDisname(:,2);

% wholeset lncrna-disease association
M = [];
for i=1:4339
    n = find(strcmp(lncname, lncrnaDisease(i,1)));
    m = find(strcmp(disname, lncrnaDisease(i,2)));
    N = [n m];
    M = [M;N];
end

nl=max(M(:,1));
nm=max(M(:,2));
pp=size(M,1);
interaction=zeros(nl,nm);
for i=1:pp
    interaction(M(i,1),M(i,2))=1;
end
wholeset_lncdismatrix = interaction;
save wholeset_lncdismatrix wholeset_lncdismatrix;

% five fold cv train data and test data
load wholeset_lncdismatrix;
lncrnaDisease = wholeset_lncdismatrix;
[r,c] = find(lncrnaDisease);
gld = [r c];   %4339*2
pp = size(gld,1);
[r0,c0] = find(lncrnaDisease==0);
fgld=[r0 c0];  
fpp = size(fgld,1);
col=sum(lncrnaDisease,1)';
row=sum(lncrnaDisease,2);

x1=randperm(pp)';
positive=gld(x1,:);
x2=randperm(fpp)';
negative=fgld(x2,:);

% 
for cv=1:3
    cv
    fprintf('cv=%d\n',cv)
    
    x1=randperm(pp)';
    positive=gld(x1,:);

    for ccv=1:5
        ccv
        %floor(pp/5)
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
            xlswrite(['wholeset_edge_f5cv',num2str(cv),'.xlsx'],edge_f1,'Sheet 1')

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
            xlswrite(['wholeset_edge_f5cv',num2str(cv),'.xlsx'],edge_f2,['Sheet ',num2str(ccv)]);

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
            xlswrite(['wholeset_edge_f5cv',num2str(cv),'.xlsx'],edge_f5,['Sheet ',num2str(ccv)])
        end

    end
    
end

%% side information

[ disease_gsSim ] = GSD( lncrnaDisease );
[ lncRNA_gsSim ] = GSM( lncrnaDisease );
[ Coslnc,CosDis] = cosSim( lncrnaDisease );

Fname2=[];
for x=1:1:9
    for y=1:1:9
            if x+y==10
               E=[x y];
               Fname2 = [Fname2;E];
            end
    end
end
F2=Fname2/10;
%LNF for lncrna similarity
sim1=lncRNA_gsSim;
sim2=Coslnc;
for i=1:9
    w=F2(i,:);
    [ sim ] = LKFtwo( w,sim1,sim2 );

    h=Fname2(i,:);
    h1=num2str(h(1));h2=num2str(h(2));
    h=strcat(h1,h2);
    simname = strcat('wholeset_lncsim',h);

    eval([simname '=sim']);
    save([simname],[simname])  
    
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
    simname = strcat('wholeset_dissim',h);
    eval([simname '=sim']);
    save([simname],[simname]) 
    
end


%% case study

% data for case study
%train data
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
%test data
x2 = randperm(fpp)';
negative = fgld(x2,:);
test = negative;   
labeltest = zeros(fpp,1);
lnc_id = test(:,1);
dis_id = test(:,2);
lnc_id = lnc_id-1;
dis_id = dis_id-1;
usaget = ones(fpp,1)*1111;
edge_f1test = [test labeltest lnc_id dis_id usaget];
edge_f11 = edge_f1test;

edge_f1 = [edge_f12;edge_f11];
xlswrite('wholeset4339casestudy.xlsx',edge_f1,'Sheet 1');

% case study result
wholesetlncrna = lncname;
wholesetdisease = disname;
predY =xlsread('PredYwholesetcasestudyresult.xlsx','sheet1');  %epoch=240,weight=0.2 0.8,0.3,0.7

length(predY)
% Number corresponding to the name
N2 = [];
for i=1:length(predY)
    
    n = wholesetlncrna(predY(i,1));
    N2 = [N2;n];
end

N = [];
for i=1:length(predY)
    
    n = wholesetdisease(predY(i,2));
    N = [N;n];
end

% predY =xlsread('PredYwholeset4339.xlsx','sheet1');  %200
% predY =xlsread('PredYwholeset4339.xlsx','sheet2');  %260
% predY =xlsread('PredYwholeset4339.xlsx','sheet3');  %200
% predY =xlsread('PredYwholeset4339.xlsx','sheet4');  %200,1919
% predY =xlsread('PredYwholeset4339.xlsx','sheet5');  %240,2837
% predY =xlsread('PredYwholeset4339.xlsx','sheet6');  %240,2819

% Top 50 lncRNAs for each disease
% clear;
% lncdisid = xlsread('wholeset2837240casestudy.xlsx','Sheet1');
% [~,lncdisname] = xlsread('wholeset2837240casestudy.xlsx','Sheet1');
% 
% lncname = lncdisname(:,5);
% disname = lncdisname(:,6);
% disid = lncdisid(:,2);
% 
% OUT2=[]
% for i=1:646
%     i
%     count=0;
%     for j = 1:54190
%         if disid(j) == i
%             count = count+1;
%         
%         end
%     end
%     OUT = [i count];
%     OUT2 = [OUT2;OUT];
% end  
% 
% OUT4=[];
% for i = 1:646
%     i
%     count=0;
%     countid=OUT2(i,2);
%     for j = 1:54190
%         if lncdisid(j,2)==i && countid~=0 && count <= countid && count <=50
%             OUT3=[lncdisid(j,1) lncdisid(j,2) lncdisid(j,3) lncdisid(j,4) lncname(j) disname(j)];
%             OUT4=[OUT4;OUT3];
%             count = count+1;
%         end
%     end
% end
% OUT4 = wholesetcasestudytop50;

% seach from three databases

% [~,lncrna] = xlsread('./ALLDATABASES3.xlsx','wholesetcasestudytop50');
% [~,ALLDatabase] = xlsread('./ALLDATABASES3.xlsx','Sheet1');
% ALLlncdis = ALLDatabase(:,1);
% allresult=length(lncrna);
% yes = {'Yes'};
% no ={'NO'};
% 
% wholeset4339_TOP50=[]
% for i=1:8000
%     i
%     flag=0;
%     for j = 1:length(ALLlncdis)
%         if strcmp(lncrna(i),ALLlncdis(j))
%             flag = 1;
%             OUT = [i yes];
%             wholeset4339_TOP50=[wholeset4339_TOP50;OUT];
%         end
%     end
%     if  flag==0
%          OUT = [i no];
%          wholeset4339_TOP50=[wholeset4339_TOP50;OUT];
%     end
% end
% 
% wholeset4339_TOP502=[]
% for i=8001:16000
%     i
%     flag=0;
%     for j = 1:length(ALLlncdis)
%         if strcmp(lncrna(i),ALLlncdis(j))
%             flag = 1;
%             OUT = [i yes];
%             wholeset4339_TOP502=[wholeset4339_TOP502;OUT];
%         end
%     end
%     if  flag==0
%          OUT = [i no];
%          wholeset4339_TOP502=[wholeset4339_TOP502;OUT];
%     end
% end
% 
% wholeset4339_TOP503=[]
% for i=16001:allresult
%     i
%     flag=0;
%     for j = 1:length(ALLlncdis)
%         if strcmp(lncrna(i),ALLlncdis(j))
%             flag = 1;
%             OUT = [i yes];
%             wholeset4339_TOP503=[wholeset4339_TOP503;OUT];
%         end
%     end
%     if  flag==0
%          OUT = [i no];
%          wholeset4339_TOP503=[wholeset4339_TOP503;OUT];
%     end
% end


