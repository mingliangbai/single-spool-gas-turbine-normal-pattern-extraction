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
inputDelays =0;%�ⲿ���������0���ӳ٣�
feedbackDelays = 1:2;%2-order PACF is significant larger than zero

trainrmse=[];valrmse=[];testrmse=[];
accab1sig=[];acctestnorsig=[];accvalnorsig=[];acctrainnorsig=[];
for hiddenLayerSize = 1:20   
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
    %% ��������Ԥ����������
    net.numInputs=4;%3���ⲿ�������ʱ����Ҫ��net.numInputs����Ϊ4
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{3}.processFcns = {'removeconstantrows','mapminmax'};
    net.inputs{4}.processFcns = {'removeconstantrows','mapminmax'};
    %% ʱ����������׼������
    [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);%������inputs,��inputSeries,targetSeries����cell��ʽ��������2��,��֪����ô���ɵ�
    %% ѵ�����ݡ���֤���ݡ��������ݻ���
    net.divideFcn = 'divideblock'; %�������ݼ���˳�򻮷����ݼ�
    net.divideMode = 'value';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    %% ����ѵ�������趨
    net.trainFcn = 'trainlm';  % Levenberg-Marquardt
    %% �����趨
    net.performFcn = 'mse';  % Mean squared error
    %% ��ͼ�����趨
    net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
        'ploterrcorr', 'plotinerrcorr'};
    %% ����ѵ��
    [net,tr] = train(net,inputs,targets,inputStates,layerStates);
    %% �������
    outputs = net(inputs,inputStates,layerStates);
    errors = gsubtract(targets,outputs);
    performance = perform(net,targets,outputs);
    %% ����ѵ��������֤�������Լ����
    trainTargets = gmultiply(targets,tr.trainMask);
    valTargets = gmultiply(targets,tr.valMask);
    testTargets = gmultiply(targets,tr.testMask);
    trainPerformance = perform(net,trainTargets,outputs);
    valPerformance = perform(net,valTargets,outputs);
    testPerformance = perform(net,testTargets,outputs);
    %% ����չʾ����open-loop narx�õ���ѵ��������֤������Լ��ϵ�ʵ��ֵ��Ԥ��ֵ
    %cell to mat
    actualtrain=cell2mat(trainTargets);
    actualval=cell2mat(valTargets);
    actualtest=cell2mat(testTargets);
    %index
    trainindex=find(isnan(actualtrain)==0);
    valindex=find(isnan(actualval)==0);
    testindex=find(isnan(actualtest)==0);
    %ʵ��ֵ
    actualtrain=actualtrain(trainindex); %fitted training data
    actualval=actualval(valindex);%fitted validation data
    actualtest=actualtest(testindex);%fitted test data  
    %Ԥ��ֵ
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
