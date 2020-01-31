clear;clc;close all
feature jit on
feature accel on
warning off
mkdir('elm_t4_156');
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

%###data normalization of normal data
%============== training set
P_train=train_input;%data1(trainrow,1:4); %获得训练输入数据
T_train=train_output;%获得训练输出数据
%==============validation set
P_val=val_input;%data1(testrow,1:4); %获得测试输入数据
T_val=val_output;%获得测试输出数据
%==============test set
P_test=test_input;%data1(testrow,1:4); %获得测试输入数据
T_test=test_output;%获得测试输出数据
%============= 归一化
% 训练集
[Pn_train,inputps] = mapminmax(P_train',-1,1);
[Tn_train,outputps] = mapminmax(T_train',-1,1);

%validation set input data normalization
Pn_val=mapminmax('apply',P_val',inputps);
% 测试集 input data normalization
Pn_test =mapminmax('apply',P_test',inputps);
%% ELM创建/训练
cd ./'elm_t4_156'
rmsetrain=[];rmseval=[];rmsetest=[];accab1box=[];accab1sig=[];accab2box=[];accab2sig=[];
acctestnorbox=[];acctestnorsig=[];accvalnorbox=[];accvalnorsig=[];acctrainbox=[];acctrainsig=[];
for hidden=1:20    
    [IW,B,LW,TF,TYPE] = elmtrain(Pn_train,Tn_train,hidden,'radbas',0);
    % 训练集效果
    Tn_train = elmpredict(Pn_train,IW,B,LW,TF,TYPE);
    T_fit = mapminmax('reverse',Tn_train,outputps);
    % ELM仿真validation
    Tn_valpre = elmpredict(Pn_val,IW,B,LW,TF,TYPE);
    % 反归一化
    T_valpre = mapminmax('reverse',Tn_valpre ,outputps);
    
    % ELM仿真测试
    Tn_sim = elmpredict(Pn_test,IW,B,LW,TF,TYPE);
    % 反归一化
    T_sim = mapminmax('reverse',Tn_sim,outputps);
    % 结果对比  【图2 测试集预测结果】
    % 均方误差
    error_test=T_sim'-T_test;
    error_train=T_fit'-T_train;
    error_val=T_valpre'-T_val;
    rmsetest=[rmsetest sqrt(mean(error_test.^2))];
    rmsetrain=[rmsetrain sqrt(mean(error_train.^2))];
    rmseval=[rmseval sqrt(mean(error_val.^2))];
    %detection
    %         accuracy(error1,error_train,1,'boxplot')
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
    pren1 = elmpredict(input1n,IW,B,LW,TF,TYPE);
    % 反归一化
    t4pre1=mapminmax('reverse',pren1,outputps);
    error1=t4pre1-output1';
    accab1sig=[accab1sig sum((abs(error1(1:end))-3*std(error_train))>0)/length(error1(1:end))];
    accab1box=[accab1box accuracy(error1,error_train,1,'boxplot')];    
    eval(['save elm_t4_156_hidden' num2str(hidden) '.mat'])
end
save elm_t4_156.mat acctrainsig accvalnorsig acctestnorsig  accab1sig rmsetrain rmseval rmsetest


