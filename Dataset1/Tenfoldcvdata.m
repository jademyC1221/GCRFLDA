% Dataset 1

%TenfoldCrossvalidation.m

load lncrnaDisease.mat;  %Dataset1 2697 lncRNA-disease association

%lncnode lncname disnode disname
load diseasesname;
load lncRNAsname;
association = cell(2697,2);
for i=1:2697
    i
    lncname=lncRNAsname{gld(i,1),1};
    disname=diseasesname{gld(i,2),1};
    association{i,1}={lncname};
    association{i,2}={disname};
end
xlswrite('Dataset1 association2697.xlsx',association);


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

% Ten fold cross-validation
for cv=1:10
    cv
    fprintf('cv=%d\n',cv)
    
    x1=randperm(pp)';
    positive=gld(x1,:);

    for ccv=1:10
        ccv
        %floor(pp/10)=269
        if ccv==1
            postest = positive(x1((ccv-1)*floor(pp/10)+1:floor(pp/10)*ccv),:);
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

            postrain = positive(x1(floor(pp/10)*ccv+1:pp),:); 
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
            xlswrite(['edge_f10cv',num2str(cv),'.xlsx'],edge_f1,'Sheet 1')

        elseif ccv<10
            postest = positive(x1((ccv-1)*floor(pp/10)+1:floor(pp/10)*ccv),:);
            numpostest = length(postest);
            x2=randperm(96183)';
            negative=fgld(x2,:);
            test = [postest;negative];
            numtest = length(test);
            labeltest = [ones(numpostest,1);zeros(fpp,1)];

            postrain1 = positive(x1(1:(ccv-1)*floor(pp/10)),:);
            postrain2 = positive(x1(ccv*floor(pp/10)+1:pp),:);
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
            xlswrite(['edge_f10cv',num2str(cv),'.xlsx'],edge_f2,['Sheet ',num2str(ccv)]);

        else
            postest = positive(x1((ccv-1)*floor(pp/10)+1:pp),:);
            x2 = randperm(fpp)';
            negative = fgld(x2,:);
            test = [postest;negative];
            numtest = length(test);
            labeltest = [ones(pp-floor(pp/10)*9,1);zeros(fpp,1)];
            lnc_id = test(:,1);
            dis_id = test(:,2);
            lnc_id = lnc_id-1;
            dis_id = dis_id-1;
            usaget = ones(numtest,1)*1111;
            edge_f10test = [test labeltest lnc_id dis_id usaget];
            x3 = randperm(numtest)';
            edge_f51 = edge_f10test(x3,:);

            postrain = positive(x1(1:(ccv-1)*floor(pp/10)),:);
            numpostrain = length(postrain);
            negtrain = fgld(x2(1:(ccv-1)*floor(pp/10)),:);
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
            xlswrite(['edge_f10cv',num2str(cv),'.xlsx'],edge_f5,['Sheet ',num2str(ccv)])
        end

    end
    
end