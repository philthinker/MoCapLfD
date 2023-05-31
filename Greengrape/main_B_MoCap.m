%main_B_MoCap
%   - Demo for Object A
%   - Unit: mm
%   - Exp. data: "Data\ObjectB_MoCap.mat"
%   
%   Haopeng Hu
%   2023.05.30


%% Data show
% - Just show the raw data.
% Raw data show: Position
%
figure;
for i = 1:8
    tmpP = DemoData(i).p(:, DemoData(i).class == 0);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(1));
    hold on;
    tmpP = DemoData(i).p(:, DemoData(i).class == 3);
    scatter3(tmpP(1,:), tmpP(2,:), tmpP(3,:),'.', 'MarkerFaceColor', Morandi_carnation(2), 'MarkerEdgeColor', Morandi_carnation(2));
    tmpP = DemoData(i).p(:, DemoData(i).class == 2);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(3));
    tmpP = DemoData(i).p(:, DemoData(i).class == 1);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(4));
end
grid on; axis equal;
xlabel('x(mm)'); ylabel('y(mm)'); zlabel('z(mm)');
view(-142, 15);
%
% Raw data show: Orientation
%
figure;
% Draw the sphere
[X,Y,Z] = sphere;
o = [0 0 0];
x = [1 0 0];
y = [0 1 0];
z = [0 0 1];
surf(X,Y,Z,'EdgeColor', 0.86*ones(1,3),'FaceColor', 0.95*ones(1,3), 'FaceAlpha', 0.48);
hold on;
quiver3(o(1),o(2),o(3),x(1), x(2), x(3), 'Color', [1.0, 0.0, 0.0], 'LineWidth', 1.8);
quiver3(o(1),o(2),o(3),y(1), y(2), y(3), 'Color', [0.0, 1.0, 0.0], 'LineWidth', 1.8);
quiver3(o(1),o(2),o(3),z(1), z(2), z(3), 'Color', [0.0, 0.0, 1.0], 'LineWidth', 1.8);
for i = 1:M
    tmpQ = DemoData(i).q(:, DemoData(i).class == 0);
    tmpP = genSphereP(tmpQ);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(1), 'LineWidth', 2.0);
    hold on;
    tmpQ = DemoData(i).q(:, DemoData(i).class == 3);
    tmpP = genSphereP(tmpQ);
    scatter3(tmpP(1,:), tmpP(2,:), tmpP(3,:),'.', 'MarkerFaceColor', Morandi_carnation(2), 'MarkerEdgeColor', Morandi_carnation(2));
    tmpQ = DemoData(i).q(:, DemoData(i).class == 2);
    tmpP = genSphereP(tmpQ);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(3), 'LineWidth', 2.0);
    tmpQ = DemoData(i).q(:, DemoData(i).class == 1);
    tmpP = genSphereP(tmpQ);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(4), 'LineWidth', 2.0);
end
grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(210, 28);
%

%% Policy learning & show
% - The policy learning method 

% Initialize and learn the policy models
model = Greengrape5S1_MoCap(DemoData, [4, 5, 8, 0.5, 1.0]);

figure;
% Demos P
for i = 1:8
    tmpC = DemoData(i).class;
    tmpLogIDN0 = tmpC ~= 0;
    tmpLogIDN3 = tmpC ~= 3;
    tmpP = DemoData(i).p(:,tmpLogIDN0 & tmpLogIDN3);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', 0.8*ones(1,3));
    hold on;
end
% Parameters
for i = 2:model.K
    % Position
    tmpP = model.p(:,i);
    tmpD = model.d(:,i) * 8;
    scatter3(tmpP(1), tmpP(2), tmpP(3), 80, colorMap(model.c(i)), 'filled');
    % Direction
    mArrow3SC(tmpP-tmpD, tmpP, 'color', colorMap(model.c(i)));
end
% Pre-Assembly
tmpP = model.p(:,1);
tmpD = model.d(:,1) * 8;
scatter3(tmpP(1), tmpP(2), tmpP(3), 80, colorMap(model.c(1)), 'filled');
mArrow3SC(tmpP, tmpP+tmpD, 'color', colorMap(model.c(1)));

tmpP0 = model.p0;
scatter3(tmpP0(1), tmpP0(2), tmpP0(3), 80, Morandi_carnation(2), 'filled');

tmpPs = [tmpP0, tmpP];
plot3(tmpPs(1,:), tmpPs(2,:), tmpPs(3,:), '--', 'Color', Morandi_carnation(2), 'LineWidth', 2.0);

grid on; axis equal;
xlabel('x(mm)'); ylabel('y(mm)'); zlabel('z(mm)');
view(-60, 38);
%}

%% Trajectory gen. 
% - Generate the pose trajectory for comparison

[cmpTraj, DemosPDTW] = model.genAssTraj_dualPolicy(DemoData);
figure;
for i = 1:8
    tmpC = DemoData(i).class;
    tmpLogIDN0 = tmpC ~= 0;
    tmpLogIDN3 = tmpC ~= 3;
    tmpP = DemoData(i).p(:,tmpLogIDN0 & tmpLogIDN3);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', 0.8*ones(1,3));
    hold on;
end
tmpP = cmpTraj(1:3,:);
plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', [1 0 0], 'LineWidth', 2.0);
grid on; axis equal;
xlabel('x(mm)'); ylabel('y(mm)'); zlabel('z(mm)');
view(-38, 38);
%

%% Exp. data show
% - Show the experiment results

pa = ExpData_CmpIV0.p(:,end)*1000;
qa = axang2quat([0 0 1 pi/3+0.11])';
qa2 = rotm2quat(axang2rotm([0 0 1 -pi/2-pi/3-0.1]))';

% Position
%
figure;
for i = 1:M
    tmpC = DemoData(i).class;
    tmpLogIDN0 = tmpC ~= 0;
    tmpLogIDN3 = tmpC ~= 3;
    tmpP = DemoData(i).p(:,tmpLogIDN0 & tmpLogIDN3);
    [tmpP, ~] = Greengrape5S1_MoCap.expTrans(pa,qa,tmpP);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', 0.70*ones(1,3));
    hold on;
end
% % Trajectory gen. by ProMP
% [tmpP, ~] = Greengrape5S1_MoCap.expTrans(pa, qa, cmpTraj(1:3,50:end));
% plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(6), 'LineWidth', 5.0);

% SAMP p
pa = [-0.8, 2.3, 0]'; qa = [1 0 0 0]'; Na = 120; %120
tmpP = ExpData_CmpIV0.p(:, ExpData_CmpIV0.phase == 3)*1000;
[tmpP1, ~] = Greengrape5S1_MoCap.expTrans(pa, qa, tmpP(:,1:Na));
tmpP2 = tmpP(:,Na+1:end);
tmpPDummy = [tmpP1, tmpP2];
plot3(tmpPDummy(1,:), tmpPDummy(2,:), tmpPDummy(3,:), 'Color', Morandi_carnation(7), 'LineWidth', 4.0);

% SAMP pg
pa = [-0.8, 2.3, 0]'; qa = [1 0 0 0]';
[tmpP, ~] = Greengrape5S1_MoCap.expTrans(pa,qa, ExpData_CmpIV0.pg(:, ExpData_CmpIV0.phase == 3)*1000);
tmpPDummy = tmpP;
for i =1:size(tmpPDummy,2)
    tmpPDummy(:,i) = tmpP(:,1) + norm(tmpP(:,i) - tmpP(:,1))*model.d(:,2);
end
plot3(tmpPDummy(1,:), tmpPDummy(2,:), tmpPDummy(3,:), 'Color', Morandi_carnation(8), 'LineWidth', 5.0);
grid on; axis equal;
xlabel('x(mm)'); ylabel('y(mm)'); zlabel('z(mm)');
view(18, 28);
%
% Orientation
%
figure;
% Draw the sphere
[X,Y,Z] = sphere;
o = [0 0 0];
x = [1 0 0];
y = [0 1 0];
z = [0 0 1];
surf(X,Y,Z,'EdgeColor', 0.86*ones(1,3),'FaceColor', 0.95*ones(1,3), 'FaceAlpha', 0.48);
hold on;
quiver3(o(1),o(2),o(3),x(1), x(2), x(3), 'Color', [1.0, 0.0, 0.0], 'LineWidth', 1.8);
quiver3(o(1),o(2),o(3),y(1), y(2), y(3), 'Color', [0.0, 1.0, 0.0], 'LineWidth', 1.8);
quiver3(o(1),o(2),o(3),z(1), z(2), z(3), 'Color', [0.0, 0.0, 1.0], 'LineWidth', 1.8);
for i = 1:M
    tmpC = DemoData(i).class;
    tmpLogIDN0 = tmpC ~= 0;
    tmpLogIDN3 = tmpC ~= 3;
    tmpQ = DemoData(i).q(:, tmpLogIDN0 & tmpLogIDN3);
    [~, tmpQ] = Greengrape5S1_MoCap.expTrans(pa,qa,zeros(3,size(tmpQ,2)), tmpQ);
    tmpP = Greengrape5S1_MoCap.genSphereP(tmpQ);
    plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', 0.86*ones(1,3), 'LineWidth', 1.2);
    hold on;
end
% % Trajectory gen. by interpolation
% [~, tmpQ] = Greengrape5S1_MoCap.expTrans(pa,qa,zeros(3,size(cmpTraj,2)), cmpTraj(4:7,:));
% tmpP = Greengrape5S1_MoCap.genSphereP(tmpQ(:,300:end));
% plot3(tmpP(1,:), tmpP(2,:), tmpP(3,:), 'Color', Morandi_carnation(6), 'LineWidth', 5.0);
% SAMP q
[~, tmpQ] = Greengrape5S1_MoCap.expTrans(pa,qa2,zeros(3,size(ExpData_CmpIV0.q,2)), ExpData_CmpIV0.q);
tmpP = Greengrape5S1_MoCap.genSphereP(tmpQ);
plot3(tmpP(1,:), tmpP(2,:), -tmpP(3,:), 'Color', Morandi_carnation(7), 'LineWidth', 4.0);
% SAMP qg
[~, tmpQ] = Greengrape5S1_MoCap.expTrans(pa,qa2,zeros(3,size(ExpData_CmpIV0.qg,2)), ExpData_CmpIV0.qg);
tmpP = Greengrape5S1_MoCap.genSphereP(tmpQ);
plot3(tmpP(1,:), tmpP(2,:), -tmpP(3,:), 'Color', Morandi_carnation(8), 'LineWidth', 4.0);

grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(-160, 18);
%


%% Functions

function [trajP] = genSphereP(trajQ, p0)
%genSphereP
%   trajQ: 4 x N, the unit quaternion
%   p0: 3 x 1, the stationary point (Default: [1 0 0]')
if nargin < 2
%     p0 = [1 0 0]';
    p0 = [1 1 1]'; p0 = p0/norm(p0);
end
N = size(trajQ,2);
trajP = repmat(p0,[1,N]);
for i = 1:N
    R = quat2rotm(trajQ(:,i)');
    trajP(:,i) = R * p0;
end
end

function [color] = colorMap(c)
%colorMap
color = 0.9*ones(1,3);
if c == 0
    % Approaching
    color = Morandi_carnation(1);
elseif c == 2
    % Insert_T
    color = Morandi_carnation(3);
elseif c == 1
    % Insert_R
    color = Morandi_carnation(4);
elseif c == 5
    % Press_Vertical
    color = Morandi_carnation(5);
end
end

