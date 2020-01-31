clear;clc;close all
a=xlsread('ELM SLFN ERNN.xlsx');
x1=1:20;
y1=a(:,3);
% �ڶ���line������
x2=1:20;
y2=a(:,2);
% ������line������
x3=1:20;
y3 = a(:,4);
%% 
figure;

% ����һ��axes
ha(1) = axes('ycolor','b','yminortick','off','xminortick','off');
hold on; % �����������ԣ���Ϊ�����bar���Զ�����axes���Ե�����
h(1) = plot(x1,y1,'color','b','marker','*','linestyle','--','linewidth',1,'markersize',8); % ����һ��barͼ
set(ha(1),'fontsize',8,'fontname','times new roman','ytick',24:0.5:26.5);
% set(h(1),'xtick',2)
xlim1 = get(ha(1),'xlim'); % ��ȡ��һ��axes��x��ķ�Χ�����ֵ�ܹؼ�
hold on
line([8],[y1(8)],'color','b','marker','o','markersize',16)
box on
grid on
set(gcf,'unit','centimeters','position',[10,10,19,8])%===%����ͼ�δ�С
%% 
% �����ڶ����ᣬҪ���һ�����غϣ�����͸������ķ�Χ��ͬ
pos1=get(ha(1),'position');
ha(2) = axes('position',pos1,'color','none','ycolor','r','yaxislocation','right','xlim',xlim1, ...
    'xtick', []);

% ����
h(2) = line(x2,y2,'color','r','parent',ha(2),'linewidth',0.26,'linestyle','-','marker','.','markersize',12);
hold on
line([3],[y2(3)],'color','r','marker','o','markersize',16)
set(ha(2),'fontsize',8,'fontname','times new roman','ytick',[2:0.5:7]);
set(gcf,'unit','centimeters','position',[10,10,19,8])%===%����ͼ�δ�С
%% 
% �����������ᣬ�����ǰ��������룬�Ҳ�Ҫ����һ���֣���˸���ķ�ΧҪ��ǰ��ķ�Χ�󣬸��ݷ�Χ����ĳ��ȳ��� % ��������
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

% ���ص���������������Ĳ���
ylim3 = get(ha(3), 'ylim');
line([xlim1(2),xlim3(2)],[ylim3(1),ylim3(1)],'parent',ha(3),'color','w');
line([xlim1(2),xlim3(2)],[ylim3(2),ylim3(2)],'parent',ha(3),'color','w');
%% 
% ��ylabel
hylab = get([ha(1);ha(2);ha(3)],'ylabel');
hxlab = get(ha(1),'xlabel');
set(hylab{1},'string','RMSE of ELM','fontsize',8,'fontname','times new roman');
set(hylab{2},'string','RMSE of SLFN','fontsize',8,'fontname','times new roman');
set(hylab{3},'string','RMSE of ERNN','fontsize',8,'fontname','times new roman');
set(ha(3),'fontsize',8,'fontname','times new roman','ytick',[2:0.5:5.5]);
% ��xlabel
set(hxlab,'string', 'Hidden neuron number','fontsize',8,'fontname','times new roman');
% set(gcf,'unit','centimeters','position',[10,10,19,6])%===%����ͼ�δ�С

% --------------------- 
% ���ߣ�_Daibingh_ 
% ��Դ��CSDN 
% ԭ�ģ�https://blog.csdn.net/healingwounds/article/details/79863136 
% ��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�


