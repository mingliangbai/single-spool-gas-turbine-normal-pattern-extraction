clear;clc;close all
feature jit on
feature accel on
warning off
%%
clear
mkdir('dataset 156 denoising');
cd ./'dataset 156 original'
load datanormal.mat
cd ..
figure
plot(p2)
hold on
plot(wden(p2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
p2=wden(p2,'heursure','s','sln',2,'sym6');

figure
plot(t1)
hold on
plot(wden(t1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t1=wden(t1,'heursure','s','sln',2,'sym6');

figure
plot(fuel)
hold on
plot(wden(fuel,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
fuel=wden(fuel,'heursure','s','sln',2,'sym6');

figure
plot(p)
hold on
plot(wden(p,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
p=wden(p,'heursure','s','sln',2,'sym6');

figure
plot(t4)
hold on
plot(wden(t4,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t4=wden(t4,'heursure','s','sln',2,'sym6');
cd ./'dataset 156 denoising'
save datanormal.mat p p2 t1 fuel t4
cd ..
%% 
clear
cd ./'dataset 156 original'
load dataab1.mat
cd ..
figure
plot(p2ab1)
hold on
plot(wden(p2ab1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
p2ab1=wden(p2ab1,'heursure','s','sln',2,'sym6');

figure
plot(t1ab1)
hold on
plot(wden(t1ab1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t1ab1=wden(t1ab1,'heursure','s','sln',2,'sym6');

figure
plot(fuelab1)
hold on
plot(wden(fuelab1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
fuelab1=wden(fuelab1,'heursure','s','sln',2,'sym6');

figure
plot(pab1)
hold on
plot(wden(pab1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
pab1=wden(pab1,'heursure','s','sln',2,'sym6');

figure
plot(t4ab1)
hold on
plot(wden(t4ab1,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t4ab1=wden(t4ab1,'heursure','s','sln',2,'sym6');
cd ./'dataset 156 denoising'
save dataab1.mat pab1 p2ab1 t1ab1 fuelab1 t4ab1
cd ..
%% 
clear
cd ./'dataset 156 original'
load dataab2.mat
cd ..
figure
plot(p2ab2)
hold on
plot(wden(p2ab2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
p2ab2=wden(p2ab2,'heursure','s','sln',2,'sym6');

figure
plot(t1ab2)
hold on
plot(wden(t1ab2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t1ab2=wden(t1ab2,'heursure','s','sln',2,'sym6');

figure
plot(fuelab2)
hold on
plot(wden(fuelab2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
fuelab2=wden(fuelab2,'heursure','s','sln',2,'sym6');

figure
plot(pab2)
hold on
plot(wden(pab2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
pab2=wden(pab2,'heursure','s','sln',2,'sym6');

figure
plot(t4ab2)
hold on
plot(wden(t4ab2,'heursure','s','sln',2,'sym6'));%heursure阈值信号处理；
legend('original','denosing1','d2')
t4ab2=wden(t4ab2,'heursure','s','sln',2,'sym6');
cd ./'dataset 156 denoising'
save dataab2.mat pab2 p2ab2 t1ab2 fuelab2 t4ab2
cd ..
