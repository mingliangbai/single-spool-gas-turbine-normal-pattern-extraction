%Elman网络
clear;clc;close all
feature jit on
feature accel on
warning off
mkdir('elman_t4_156');
%%
cd ..
addpath('./toolbox')
cd ./'data'
load datanormal.mat
load dataab1.mat
cd ..
cd ./'ELM and other ANN methods'
%###############data division
trainindex=1:round(0.7*length(t4));
valindex=(round(0.7*length(t4))+1):round(0.85*length(t4));
testindex=(round(0.85*length(t4))+1):length(t4);
train_input=[fuel(trainindex) t1(trainindex) p2(trainindex)];
train_output=t4(trainindex);
val_input=[fuel(valindex) t1(valindex) p2(valindex)];
val_output=t4(valindex);
test_input=[fuel(testindex) t1(testindex) p2(testindex)];
test_output=t4(testindex);

%选连样本输入输出数据归一化
[inputn_train,inputps]=mapminmax(train_input');
[outputn,outputps]=mapminmax(train_output');
inputn_val=mapminmax('apply',val_input',inputps);
inputn_test=mapminmax('apply',test_input',inputps);
%% elman模型创建/训练
% 利用循环，设置不同的隐藏层神经元个数
cd ./'elman_t4_156'
rmsetrain=[];rmseval=[];rmsetest=[];accab1box=[];accab1sig=[];accab2box=[];accab2sig=[];
acctestnorbox=[];acctestnorsig=[];accvalnorbox=[];accvalnorsig=[];acctrainbox=[];acctrainsig=[];
for hidden=1:20
    threshold=repmat([-1,1],3,1);
    % 建立Elman神经网络 隐藏层为nn(i)个神经元
    net=newelm(threshold,[hidden,1],{'tansig','purelin'});
    % 设置网络训练参数
    net.trainparam.epochs=200;
    % 初始化网络
    net=init(net);
    % Elman网络训练
    net=train(net,inputn_train,outputn);
    %%     predict
    trainpre0=sim(net,inputn_train);
    valpre0=sim(net,inputn_val);
    testpre0=sim(net,inputn_test);
    %网络输出反归一化
    trainpre=mapminmax('reverse',trainpre0,outputps);
    valpre=mapminmax('reverse',valpre0,outputps);
    testpre=mapminmax('reverse',testpre0,outputps);
    error_train=trainpre'-train_output;
    error_val=valpre'-val_output;
    error_test=testpre'-test_output;
    rmsetest=[rmsetest sqrt(mean(error_test.^2))];
    rmsetrain=[rmsetrain sqrt(mean(error_train.^2))];
    rmseval=[rmseval sqrt(mean(error_val.^2))];
    acctestnorbox=[acctestnorbox accuracy(error_test,error_train,0,'boxplot')];
    accvalnorbox=[accvalnorbox accuracy(error_val,error_train,0,'boxplot')];
    acctrainbox=[acctrainbox accuracy(error_train,error_train,0,'boxplot')];
    acctestnorsig=[acctestnorsig sum((abs(error_test(1:end))-3*std(error_train))<0)/length(error_test(1:end))];
    accvalnorsig=[accvalnorsig sum((abs(error_val(1:end))-3*std(error_train))<0)/length(error_val(1:end))];
    acctrainsig=[acctrainsig sum((abs(error_train(1:end))-3*std(error_train))<0)/length(error_train(1:end))];
    %% anomaly detection for dataset1
    input1=[fuelab1 t1ab1 p2ab1];
    output1=t4ab1;
    input1n=mapminmax('apply',input1',inputps);
    input1pren = sim(net,input1n);%elmpredict(input1n,IW,B,LW,TF,TYPE);
    % 反归一化
    t4pre1=mapminmax('reverse',input1pren,outputps);
    error1=t4pre1-output1';
    accab1sig=[accab1sig sum((abs(error1(1:end))-3*std(error_train))>0)/length(error1(1:end))];
    eval(['save elman_t4_156_hidden' num2str(hidden)  '.mat'])    
end
save elman_t4_156.mat acctrainsig accvalnorsig acctestnorsig  accab1sig rmsetrain rmseval rmsetest

