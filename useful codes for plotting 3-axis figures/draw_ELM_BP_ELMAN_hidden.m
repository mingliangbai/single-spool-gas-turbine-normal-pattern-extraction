clear;clc;close all
a=xlsread('ELM SLFN ERNN.xlsx');
x1=1:20;
y1=a(:,3);
% 第二个line的数据
x2=1:20;
y2=a(:,2);
% 第三个line的数据
x3=1:20;
y3 = a(:,4);
%% 
figure;

% 建第一个axes
ha(1) = axes('ycolor','b','yminortick','off','xminortick','off');
hold on; % 保持以上属性，因为下面的bar有自动调整axes属性的作用
h(1) = plot(x1,y1,'color','b','marker','*','linestyle','--','linewidth',1,'markersize',8); % 画第一个bar图
set(ha(1),'fontsize',8,'fontname','times new roman','ytick',24:0.5:26.5);
% set(h(1),'xtick',2)
xlim1 = get(ha(1),'xlim'); % 获取第一个axes的x轴的范围，这个值很关键
hold on
line([8],[y1(8)],'color','b','marker','o','markersize',16)
box on
grid on
set(gcf,'unit','centimeters','position',[10,10,19,8])%===%设置图形大小
%% 
% 建立第二个轴，要与第一个轴重合，并且透明，轴的范围相同
pos1=get(ha(1),'position');
ha(2) = axes('position',pos1,'color','none','ycolor','r','yaxislocation','right','xlim',xlim1, ...
    'xtick', []);

% 画线
h(2) = line(x2,y2,'color','r','parent',ha(2),'linewidth',0.26,'linestyle','-','marker','.','markersize',12);
hold on
line([3],[y2(3)],'color','r','marker','o','markersize',16)
set(ha(2),'fontsize',8,'fontname','times new roman','ytick',[2:0.5:7]);
set(gcf,'unit','centimeters','position',[10,10,19,8])%===%设置图形大小
%% 
% 建立第三个轴，左侧与前两个轴对齐，右侧要长出一部分，因此该轴的范围要比前轴的范围大，根据范围与轴的长度成正 % 比来计算
pos1(1)=pos1(1)-0.02;
pos1(3) = pos1(3)*.86;
set([ha(1);ha(2)],'position',pos1);
pos3 = pos1;
pos3(3) = pos3(3)+.12;
xlim3 = xlim1;
xlim3(2) = xlim3(1)+(xlim1(2)-xlim1(1))/pos1(3)*pos3(3);
ha(3) = axes('position',pos3, 'color','none','ycolor',[1,0,1],'xlim',xlim3, ...
    'xtick',[],'yaxislocation','right','yminortick','off');
%% 
[x3, ind] = sort(x3, 2, 'ascend');
y3 = y3(ind);
ind2 =  (x3<=xlim1(2));
y3 = y3(ind2);
x3 = x3(ind2);
h(3) = line(x3, y3,'color',[1,0,1],'linewidth',0.25,'parent',ha(3),'marker','s','markersize',8);
hold on
line([10],[y3(10)],'color',[1,0,1],'marker','o','markersize',16)

% set(gcf,'unit','centimeters','position',[10,10,19,6])

% 隐藏第三个横轴伸出来的部分
ylim3 = get(ha(3), 'ylim');
line([xlim1(2),xlim3(2)],[ylim3(1),ylim3(1)],'parent',ha(3),'color','w');
line([xlim1(2),xlim3(2)],[ylim3(2),ylim3(2)],'parent',ha(3),'color','w');
%% 
% 加ylabel
hylab = get([ha(1);ha(2);ha(3)],'ylabel');
hxlab = get(ha(1),'xlabel');
set(hylab{1},'string','RMSE of ELM','fontsize',8,'fontname','times new roman');
set(hylab{2},'string','RMSE of SLFN','fontsize',8,'fontname','times new roman');
set(hylab{3},'string','RMSE of ERNN','fontsize',8,'fontname','times new roman');
set(ha(3),'fontsize',8,'fontname','times new roman','ytick',[2:0.5:5.5]);
% 加xlabel
set(hxlab,'string', 'Hidden neuron number','fontsize',8,'fontname','times new roman');
% set(gcf,'unit','centimeters','position',[10,10,19,6])%===%设置图形大小

% --------------------- 
% 作者：_Daibingh_ 
% 来源：CSDN 
% 原文：https://blog.csdn.net/healingwounds/article/details/79863136 
% 版权声明：本文为博主原创文章，转载请附上博文链接！


