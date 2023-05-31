function [assTraj, DemosA3PDTW] = genAssTraj_dualPolicy(obj, DemoData, N, WinDTW, Ks, hs)
%genAssTraj Generate the trajectory in assembling phase.
%   DemoData: 1 x M, the DemoData construct by obj.constructDemoData().
%   N; Integer >0, the num. of data in the traj. (Default: 1000)
%   WinDTW: Integer >0, the window of DTW. (Default: 30)
%   Ks: Integer >0, the number of kernels used. (Default: [15,15])
%   hs: Scalar >0, the width of the kernels. (Default: [0.001,0.001]);
%   -------------------------------------------------
%   assTraj: 7 x N, the assembling trajectory. [p;q]
%   DemosA3PDTW: 1 x M cell of 7 x N [i,p] data, the demo data in A3 after DTW.
%   @Greengrape5S1_MoCap


M = length(DemoData);
if nargin < 3 || isempty(N)
    N = 1000;
end

%% Init. data
DemosA3 = cell(1,M);    % p;q
pa = obj.p(:,1);
for i = 1:M
    % Figure out the assembling data
    tmpC = DemoData(i).class;
    tmpLogIDN0 = tmpC ~= 0;
    tmpLogIDN3 = tmpC ~= 3;
    tmpP = DemoData(i).p(:,tmpLogIDN0 & tmpLogIDN3);
    tmpQ = DemoData(i).q(:,tmpLogIDN0 & tmpLogIDN3);
    % Get rid of the outliers
    [~, tmpID] = min(diag((tmpP - pa)'*(tmpP - pa)));
    DemosA3{i} = [tmpP(:,tmpID:end); tmpQ(:, tmpID:end)];
end

%% DTW

if nargin < 4 || isempty(WinDTW)
    WinDTW = 30;
end
[DemosA3PDTW] = initDemos(DemosA3, WinDTW);

%% ProMP for position

tmpDemos = [];
tmpDemos.data = [];
tmpDemos = repmat(tmpDemos,[1,M]);
for i = 1:M
    tmpData = DemosA3PDTW{i};
    tmpDemos(i).data = tmpData(2:end,:);
end

if nargin < 5 || isempty(Ks)
    Ks = [15,15];
end
if nargin < 6 || isempty(hs)
    hs = [0.001,0.001];
end
modelP = ProMPZero(3,Ks(1),hs(1));
modelP = modelP.learnLRR(tmpDemos);
trajP = modelP.reproduct(N);

%% Orientation 
% figure;
% for i = 1:4
%     subplot(4,1,i);
%     for j = 1:M
%         tmpQ = DemosA3{j}; tmpQ = tmpQ(4:7,:);
%         t = (1:size(tmpQ,2));
%         plot(t, tmpQ(i,:));
%         hold on;
%     end
%     grid on;
% end

% Interpolation
trajQ = DemosA3{M}; trajQ = trajQ(4:7,:);
tmpX = linspace(0,1,size(trajQ,2));
trajQ = vNormalize(pchip(tmpX, trajQ, linspace(0,1,N)));

assTraj = [trajP;trajQ];

end

function [DemosP] = initDemos(DemosA3, DTW_Window)
%initDemos
M = length(DemosA3);
DemosP = cell(1,M);
% DemosEta = cell(1,M);
for i = 1:M
    tmpData = DemosA3{i};
    tmpP = tmpData(1:3,:);
%     tmpQ = tmpData(7:10,:);
%     tmpEta = quatLogMap(tmpQ, qa);
    DemosP{i} = tmpP;
%     tmpPhase = diag((tmpP - pa)' * (tmpP - pa))';
%     DemosEta{i} = [tmpPhase; tmpEta];
end
DemosP = GreengrapeDTW(DemosP, DTW_Window);
for i = 1:M
    tmpData = DemosP{i};
    N = size(tmpData, 2);
    DemosP{i} = [linspace(0,1,N); tmpData];
end
end

function [DemosDTW] = GreengrapeDTW(Demos, Window)
%GreengrapeDTW
M = length(Demos);
tmpDemos = cell(1,M);
for i = 1:M
    tmpData = Demos{i};
    tmpDemos{i} = tmpData';
end
DemosDTW = iceDTW(tmpDemos, Window);
for i = 1:M
    tmpData = DemosDTW{i};
    DemosDTW{i} = tmpData';
end
end
