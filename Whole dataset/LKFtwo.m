function [ sim ] = LKFtwo( w,sim1,sim2 )
sim = w(1)*sim1 + w(2)*sim2;
end