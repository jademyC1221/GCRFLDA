%% case study
load lncrnaDisease.mat;    %Dataset1 2697 lncRNA-disease association
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
xlswrite(['edge2697casestudy.xlsx'],edge_f1,'Sheet 1');
