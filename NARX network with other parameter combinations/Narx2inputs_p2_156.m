%% 加载数据
clear;clc;close all
feature jit on
feature accel on
warning off
mkdir('narx_p2_156');
cd ..
addpath('./toolbox')
cd ./'data'
load datanormal.mat
load dataab1.mat
cd ..
cd ./'NARX network with other parameter combinations'/'narx_p2_156'
egtsmooth=p2;
%% 建立非线性自回归模型
close all
inputSeries=num2cell([fuel';t1']);%two external inputs, fuel and t1 are column vectors of the same length
targetSeries=num2cell(egtsmooth');%NWP的预测误差
inputDelays =1:1;%外部输入变量1的1阶延迟；
%如何给外部输入变量1和外部输入变量2设置不同的延迟阶数还没弄明白
fdelay=2;%5-order PACF is significant larger than zero
feedbackDelays = 1:fdelay;%自回归的fdelay阶延迟

%%%%%%%%%
trainrmse=[];trainmae=[];trainmape=[]; valrmse=[];valmae=[];valmape=[];testmae=[];testmape=[];testrmse=[];
accab1box=[];accab1sig=[];
acctestnorbox=[];acctestnorsig=[];accvalnorbox=[];accvalnorsig=[];acctrainnorbox=[];acctrainnorsig=[];
for hiddenLayerSize = 1:2
        %eval(['cd ./narx_hidden_' num2str(hiddenLayerSize)])
        net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
        %% 网络数据预处理函数定义,这两句不用预先输入数据,就是给一个规则挂上
        net.numInputs=3;%两个外部输入变量时，将net.numInputs设置为2+1=3即可进行正常的训练，net.numInputs比实际的Narx网络输入多1
        net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};
        net.inputs{3}.processFcns = {'removeconstantrows','mapminmax'};
        net.trainParam.epochs=50;
        net.trainParam.lr=0.1;
         net.trainParam.mc=0.01;%动量因子
        net.trainParam.goal=0.0000004;
        % net.inputs{2}.feedbackOutput=[];
        %% 时间序列数据准备工作
        [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);%生成了inputs,把inputSeries,targetSeries两个cell格式的摞成了2行,不知道怎么生成的
        %% 训练数据、验证数据、测试数据划分
        net.divideFcn = 'divideblock'; %按照数据集的顺序划分数据集，前trainRatio 为 训练集；后testRatio的数据为测试集  时间序列数据应采用该方法进行划分数据集，可以open divideblock查看具体信息
        %此外还有divideind, divideint, dividerand, dividetrain.几种划分数据集的方式;
        %dividerand随机划分数据集，在不需要考虑时间序列或数据时序特性不明显时可采用随机划分数据集的方式
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
        %    perform(NET,T,Y,EW) takes a network, targets T and outputs Y, and optionally error weights EW, and returns performance using
        %    the network's default performance function NET.performFcn.
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
        
        predictoutputs=cell2mat(outputs);
        %预测值
        fittrain=predictoutputs(trainindex);%actual training data
        fitval=predictoutputs(valindex);%actual validation data
        fittest=predictoutputs(testindex);%actual test data
        %% 网络训练效果可视化
%         figure; plot(fittrain);hold on;plot(actualtrain);legend('fitted','actual');title('training data')
%         figure; plot(fitval);hold on;plot(actualval);legend('fitted','actual');title('validation data')
%         figure; plot(fittest,'b');hold on;plot(actualtest,'r');legend('fitted','actual');title('test data')
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
        %% 异常检测1
        %         cd ..        
        inputSeries3=num2cell([fuelab1';t1ab1']);%two external inputs, fuel and t1 are column vectors of the same length
        targetSeries3=num2cell(p2ab1');%NWP的预测误差
        [inputs3,inputStates3,layerStates3,targets3] = preparets(net,inputSeries3,{},targetSeries3);%生成了inputs,把inputSeries,targetSeries两个cell格式的摞成了2行,不知道怎么生成的
        outputs3 = net(inputs3,inputStates3,layerStates3);
        errors3 = gsubtract(targets3,outputs3);
        accab1sig=[accab1sig sum(abs(cell2mat(errors3))-3*std(trainerror)>0)/length(cell2mat(errors3))];
        accab1box=[accab1box accuracy(cell2mat(errors3),trainerror,1,'boxplot')];
        eval(['save narx_p2_hidden' num2str(hiddenLayerSize) '.mat'])
end
%%
save narx_p2_156.mat acctrainnorbox accvalnorbox acctestnorbox  accab1box acctrainnorsig accvalnorsig acctestnorsig  accab1sig trainmae trainmape valrmse valmae valmape testmae testmape testrmse
cd ..
