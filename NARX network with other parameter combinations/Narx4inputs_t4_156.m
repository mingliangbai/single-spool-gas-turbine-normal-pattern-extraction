%% ��������
clear;clc;close all
feature jit on
feature accel on
warning off
mkdir('Narx4inputs_t4_156');
cd ..
addpath('./toolbox')
cd ./'data'
load datanormal.mat
load dataab1.mat
cd ..
cd ./'NARX network with other parameter combinations'/'Narx4inputs_t4_156'
egtsmooth=t4;
%% �����������Իع�ģ��
close all
inputSeries=num2cell([fuel';t1';p2';p']);%two external inputs, fuel and t1 are column vectors of the same length
targetSeries=num2cell(egtsmooth');%NWP��Ԥ�����
inputDelays =0;%�ⲿ�������1��0���ӳ٣�
%��θ��ⲿ�������1���ⲿ�������2���ò�ͬ���ӳٽ�����ûŪ����
fdelay=2;%5-order PACF is significant larger than zero
feedbackDelays = 1:fdelay;%�Իع��fdelay���ӳ�

%%%%%%%%%
trainrmse=[];trainmae=[];trainmape=[]; valrmse=[];valmae=[];valmape=[];testmae=[];testmape=[];testrmse=[];
accab1box=[];accab1sig=[];accab2box=[];accab2sig=[];
acctestnorbox=[];acctestnorsig=[];accvalnorbox=[];accvalnorsig=[];acctrainnorbox=[];acctrainnorsig=[];
for hiddenLayerSize =1:20
        net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
        %% ��������Ԥ������������,�����䲻��Ԥ����������,���Ǹ�һ���������
        net.numInputs=5;%3���ⲿ�������ʱ����net.numInputs����Ϊ3+1=4���ɽ���������ѵ����net.numInputs��ʵ�ʵ�Narx���������1
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{3}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{4}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{5}.processFcns = {'removeconstantrows','mapminmax'};
        % net.inputs{2}.feedbackOutput=[];
        %% ʱ����������׼������
        [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);%������inputs,��inputSeries,targetSeries����cell��ʽ��������2��,��֪����ô���ɵ�
        %% ѵ�����ݡ���֤���ݡ��������ݻ���
        net.divideFcn = 'divideblock'; %�������ݼ���˳�򻮷����ݼ���ǰtrainRatio Ϊ ѵ��������testRatio������Ϊ���Լ�  ʱ����������Ӧ���ø÷������л������ݼ�������open divideblock�鿴������Ϣ
        %���⻹��divideind, divideint, dividerand, dividetrain.���ֻ������ݼ��ķ�ʽ;
        %dividerand����������ݼ����ڲ���Ҫ����ʱ�����л�����ʱ�����Բ�����ʱ�ɲ�������������ݼ��ķ�ʽ
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
        
        predictoutputs=cell2mat(outputs);
        %Ԥ��ֵ
        fittrain=predictoutputs(trainindex);%actual training data
        fitval=predictoutputs(valindex);%actual validation data
        fittest=predictoutputs(testindex);%actual test data
        %% ����ѵ��Ч�����ӻ�
        trainerror=actualtrain-fittrain;
        valerror=actualval-fitval;
        testerror=actualtest-fittest;
        
        trainrmse=[trainrmse sqrt(mean(trainerror.^2))];
        valrmse=[valrmse sqrt(mean(valerror.^2))];
        testrmse=[testrmse sqrt(mean(testerror.^2))];
        
        trainmae=[trainmae mean(abs(trainerror))];
        valmae=[valmae mean(abs(valerror))];
        testmae=[testmae mean(abs(testerror))];
        
        trainmape=[trainmape mean(abs(trainerror./actualtrain))];
        valmape=[valmape mean(abs(valerror./actualval))];
        testmape=[testmape mean(abs(testerror./actualtest))];
        
        acctestnorsig=[acctestnorsig sum((abs(testerror(1:end))-3*std(trainerror))<0)/length(testerror(1:end))];%
        acctestnorbox=[acctestnorbox accuracy(testerror,trainerror,0,'boxplot')];
        accvalnorsig=[accvalnorsig sum((abs(valerror)-3*std(trainerror))<0)/length(valerror)];%
        accvalnorbox=[accvalnorbox accuracy(valerror,trainerror,0,'boxplot')];
        acctrainnorsig=[acctrainnorsig accuracy(trainerror,trainerror,0,'3sigma')];
        acctrainnorbox=[acctrainnorbox accuracy(trainerror,trainerror,0,'boxplot')];
        %% �쳣���1
        %         cd ..        
        inputSeries3=num2cell([fuelab1';t1ab1';p2ab1';pab1']);%two external inputs, fuel and t1 are column vectors of the same length
        targetSeries3=num2cell(t4ab1');%NWP��Ԥ�����
        [inputs3,inputStates3,layerStates3,targets3] = preparets(net,inputSeries3,{},targetSeries3);%������inputs,��inputSeries,targetSeries����cell��ʽ��������2��,��֪����ô���ɵ�
        outputs3 = net(inputs3,inputStates3,layerStates3);
        errors3 = gsubtract(targets3,outputs3);
        accab1sig=[accab1sig sum(abs(cell2mat(errors3))-3*std(trainerror)>0)/length(cell2mat(errors3))];
        accab1box=[accab1box accuracy(cell2mat(errors3),trainerror,1,'boxplot')];
        eval(['save narx_t4_hidden' num2str(hiddenLayerSize)  '.mat'])
end
%%
save narx_t4_156.mat acctrainnorbox accvalnorbox acctestnorbox  accab1box  acctrainnorsig accvalnorsig acctestnorsig  accab1sig  trainmae trainmape valrmse valmae valmape testmae testmape testrmse
cd ..