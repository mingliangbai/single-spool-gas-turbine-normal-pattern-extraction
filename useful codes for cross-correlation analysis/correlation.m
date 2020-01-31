%% autocorr     cross-correlation
clear
cd ..
addpath('./toolbox')
cd ./'data'
load datanormal
cd ..
close all
figure
autocorr(t4(1:16136))%round(23052*0.7)=16136, carry out correlation analysis for training data
grid on
xlabel('Lag','fontsize',8,'FontWeight','bold')
ylabel('ACF','fontsize',8,'FontWeight','bold')
set(gca,'fontname','times new roman','fontsize',8,'FontWeight','bold')
set(gcf,'unit','centimeters','position',[10,10,8,6])
title('')
% bar(a1)
% xbins=0:1:20;
% % ybins=2:0.2:3;
% set(gca,'XTickLabel',xbins);
% % set(gca,'YTickLabel',ybins); 
figure
parcorr(t4(1:16136))
grid on
xlabel('Lag','fontsize',8,'FontWeight','bold')
ylabel('PACF','fontsize',8,'FontWeight','bold')
set(gca,'fontname','times new roman','fontsize',8,'FontWeight','bold')
set(gcf,'unit','centimeters','position',[10,10,8,6])
title('')
%################
figure
subplot(1,3,1)
a=mycrosscorr(t1(1:16136),t4(1:16136),20);
grid on
xlabel('Lag of t_1','fontname','times new roman','fontsize',8,'FontWeight','bold')
ylabel('correlation coefficient','fontname','times new roman','fontsize',8,'FontWeight','bold')
set(gca,'fontname','times new roman','fontsize',8,'FontWeight','bold')
subplot(1,3,2)
b=mycrosscorr(fuel(1:16136),t4(1:16136),20);
grid on
xlabel('Lag of g_{fuel}','fontsize',8,'FontWeight','bold')
set(gca,'fontname','times new roman','fontsize',8,'FontWeight','bold')
% ylabel('correlation coefficient','fontsize',8,'FontWeight','bold')
subplot(1,3,3)
c=mycrosscorr(p2(1:16136),t4(1:16136),20);
grid on
xlabel('Lag of p_2','fontname','times new roman','fontsize',8,'FontWeight','bold')
% ylabel('correlation coefficient','fontsize',8,'FontWeight','bold')
set(gca,'fontname','times new roman','fontsize',8,'FontWeight','bold')
set(gcf,'unit','centimeters','position',[10,10,19,6])