%% load data
clear;clc;close all
addpath('./toolbox')
feature jit on
feature accel on
warning off
mkdir('narx_3inputs_t4_156');
cd ./'data'
load datanormal.mat
load dataab1.mat
cd ..
cd ./'narx_3inputs_t4_156'
egtsmooth=t4;
%% establish NARX model
close all
inputSeries=num2cell([fuel';t1';p2']);
targetSeries=num2cell(egtsmooth');
inputDelays =0;%外部输入变量的0阶延迟；
feedbackDelays = 1:2;%2-order PACF is significant larger than zero

trainrmse=[];valrmse=[];testrmse=[];
accab1sig=[];acctestnorsig=[];accvalnorsig=[];acctrainnorsig=[];
for hiddenLayerSize = 1:20   
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
    %% 网络数据预处理函数定义
    net.numInputs=4;%3个外部输入变量时，需要将net.numInputs设置为4
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{3}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{4}.processFcns = {'removeconstantrows','mapminmax'};
    %% 时间序列数据准备工作
    [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);%生成了inputs,把inputSeries,targetSeries两个cell格式的摞成了2行,不知道怎么生成的
    %% 训练数据、验证数据、测试数据划分
    net.divideFcn = 'divideblock'; %按照数据集的顺序划分数据集
    net.divideMode = 'value';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    %% 网络训练函数设定
    net.trainFcn = 'trainlm';  % Levenberg-Marquardt
    %% 误差函数设定
    net.performFcn = 'mse';  % Mean squared error
    %% 绘图函数设定
    net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
        'ploterrcorr', 'plotinerrcorr'};
    %% 网络训练
    [net,tr] = train(net,inputs,targets,inputStates,layerStates);
    %% 网络测试
    outputs = net(inputs,inputStates,layerStates);
    errors = gsubtract(targets,outputs);
    performance = perform(net,targets,outputs);
    %% 计算训练集、验证集、测试集误差
    trainTargets = gmultiply(targets,tr.trainMask);
    valTargets = gmultiply(targets,tr.valMask);
    testTargets = gmultiply(targets,tr.testMask);
    trainPerformance = perform(net,trainTargets,outputs);
    valPerformance = perform(net,valTargets,outputs);
    testPerformance = perform(net,testTargets,outputs);
    %% 具体展示利用open-loop narx得到的训练集、验证集与测试集上的实际值及预测值
    %cell to mat
    actualtrain=cell2mat(trainTargets);
    actualval=cell2mat(valTargets);
    actualtest=cell2mat(testTargets);
    %index
    trainindex=find(isnan(actualtrain)==0);
    valindex=find(isnan(actualval)==0);
    testindex=find(isnan(actualtest)==0);
    %实际值
    actualtrain=actualtrain(trainindex); %fitted training data
    actualval=actualval(valindex);%fitted validation data
    actualtest=actualtest(testindex);%fitted test data  
    %预测值
    predictoutputs=cell2mat(outputs);
    fittrain=predictoutputs(trainindex);%actual training data
    fitval=predictoutputs(valindex);%actual validation data
    fittest=predictoutputs(testindex);%actual test data
    %% evalauting performance
    %reiduals
    trainerror=actualtrain-fittrain;
    valerror=actualval-fitval;
    testerror=actualtest-fittest;
    %rmse
    trainrmse=[trainrmse sqrt(mean(trainerror.^2))];
    valrmse=[valrmse sqrt(mean(valerror.^2))];
    testrmse=[testrmse sqrt(mean(testerror.^2))];    
    %accuracy
    acctestnorsig=[acctestnorsig sum((abs(testerror(1:end))-3*std(trainerror))<0)/length(testerror(1:end))];%
    accvalnorsig=[accvalnorsig sum((abs(valerror)-3*std(trainerror))<0)/length(valerror)];%
    acctrainnorsig=[acctrainnorsig accuracy(trainerror,trainerror,0,'3sigma')];
    %% anomaly detection
    inputSeries3=num2cell([fuelab1';t1ab1';p2ab1']);
    targetSeries3=num2cell(t4ab1');
    [inputs3,inputStates3,layerStates3,targets3] = preparets(net,inputSeries3,{},targetSeries3);
    outputs3 = net(inputs3,inputStates3,layerStates3);
    errors3 = gsubtract(targets3,outputs3);
    accab1sig=[accab1sig sum(abs(cell2mat(errors3))-3*std(trainerror)>0)/length(cell2mat(errors3))];    
    eval(['save narx_t4_hidden' num2str(hiddenLayerSize)  '.mat'])
end
%%
save narx_t4_156.mat acctrainnorsig accvalnorsig acctestnorsig  accab1sig  trainrmse valrmse testrmse
cd ..
