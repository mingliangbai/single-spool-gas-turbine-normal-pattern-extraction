function [acc,da,xiao]=accuracy(testerror,trainerror,datatype,method)
% acc=accuracy(testerror,trainerror,datatype,method)
% testerror:error of test data
% trainerror: error of training data
% datatype: 0 represents normal data; 1 represents abnormal data
% method: '3sigma' represents threshold based on 3 sigma principle;
% 'boxplot' represents boxplot threshold;
switch method
    case '3sigma'
        xiao=-3*std(trainerror);
        da=3*std(trainerror);
    case 'boxplot'
        Q1=quantile(trainerror,0.25);
        Q3=quantile(trainerror,0.75);
        IQR=Q3-Q1;
        xiao=Q1-1.5*IQR;
        da=Q3+1.5*IQR;
end
switch datatype
    case 0%normal ,  xiao<testerror<da的样本分类正确
        acc=1-(sum(testerror>da)+sum(testerror<xiao))/length(testerror);
    case 1%abnormal ,  testerror<xiao或testerror>da的样本分类正确
        acc=(sum(testerror>da)+sum(testerror<xiao))/length(testerror);
end
end