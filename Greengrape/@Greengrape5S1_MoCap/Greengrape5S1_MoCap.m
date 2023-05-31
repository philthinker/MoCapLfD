classdef Greengrape5S1_MoCap
    %Greengrape5S1_MoCap MoCap version of @Greengrape 5S.1.0
    %   - Initial the object with the recogition result.
    %   - There is temporally NO voting strategy considered.
    %   - The unit of length is mm.
    %   - The DP algorithm only works for Insert_T and is run once only.
    %   - We assume that the MPs of the same type must be consecutive.
    %
    %   Haopeng Hu
    %   2022.10.15
    %   All rights reserved
    %
    %   Classficiations:
    %   |   0: Approaching
    %   |   1: Insert_R
    %   |   2: Insert_T
    %   |   3: Preparation
    %   |   4: Press_Horizental
    %   |   5: Press_Vertical
    
    properties (Access = public)
        K;      % Integer, the number of MPs+1
        p0;    % 3 x 1, the pre-alignment position
        p;      % 3 x K, the desired positions
        q;      % 4 x K, the desired orientations
        d;      % 3 x K, the desired directions
        c;      % 1 x K, the classifications (0 for pre-assembly)
    end
    
    properties (Access = private)
        param_l = 4;                % Scalar >0, the alignment length.
        param_thdDP = 5;        % Scalar >0, the DP threshold.
        param_corr0 = 3;          % Scalar >0, correction of the pre-assembly position in the desired direction
        param_corrx = -0.5;      % Scalar >0, correction of the pre-assembly position in x-axis
        param_corry = 0.5;        % Scalar >0, correction of the pre-assembly position in y-axis
        param_maxK = 10;        % Integer >0, the max. K.
    end
    
    methods
        function obj = Greengrape5S1_MoCap(DemoData, params, Rl)
            %Greengrape5S1_MoCap Initial the object with the recogition result.
            %   DemoData: 1 x M, the DemoData construct by obj.constructDemoData().
            %   params: 1 x ?, the hyper-parameters. (Default: [])
            %   Rl: 3 x 3 SO(3), the alignment direction correction coefficient.
            obj.K = 0;
            obj.p = []; obj.q = []; obj.d = []; obj.c = []; obj.p0 = [];
            if nargin < 1 || isempty(DemoData)
                return;
            end
            if nargin >= 2
                obj.param_l = params(1);
                obj.param_thdDP = params(2);
                obj.param_corr0 = params(3);
                obj.param_corrx = params(4);
                obj.param_corry = params(5);
            end
            if nargin < 3
                Rl = eye(3);
            end
            %% Learning algorithm
            M = length(DemoData);
            obj.K = 1;
            tmp_c = zeros(1,obj.param_maxK);
            %% Find the num. and seq. of MPs preliminarily
            curr_c = 0;
            tmpC = DemoData(1).class;
            tmpLogID = tmpC ~= 3;
            tmpC = tmpC(tmpLogID);
            for j = 1:length(tmpC)
                if tmpC(j) ~= curr_c
                    obj.K = obj.K + 1;
                    tmp_c(obj.K) = tmpC(j);
                    curr_c = tmpC(j);
                end
            end
            tmp_c = tmp_c(1:obj.K);
            %% Learn the parameters preliminarily
            obj.c = tmp_c;
            tmp_p = zeros(3,M,obj.K);
            tmp_q = repmat([1 0 0 0]', [1,M,obj.K]);
            tmp_d = zeros(3,M,obj.K);
            for i = 1:M
                tmpLogID = DemoData(i).class ~= 3;
                tmpC = DemoData(i).class(tmpLogID);
                tmpP = DemoData(i).p(:, tmpLogID);
                tmpQ = DemoData(i).q(:, tmpLogID);
                for j = 1:obj.K
                    curr_c = obj.c(j);
                    currP = tmpP(:,tmpC == curr_c);
                    currQ = tmpQ(:,tmpC == curr_c);
                    tmp_p(:,i,j) = currP(:,end);
                    tmp_q(:,i,j) = currQ(:,end);
                    if curr_c == 1
                        % Insert_R
                        tmp_d(:,i,j) = vNormalize(quatLogMap(currQ(:,end),currQ(:,1)));
                    elseif curr_c == 2
                        % Insert_T
                        tmp_d(:,i,j) = vNormalize(currP(:,end) - currP(:,1));
                    elseif curr_c == 5
                        % Press_Vertical
                        tmp_d(:,i,j) = vNormalize(currP(:,end) - currP(:,1));
                    end
                end
            end
            for j = 1:obj.K
                obj.p(:,j) = mean(tmp_p(:,:,j),2);
                obj.q(:,j) = quatAverage(tmp_q(:,:,j));
                obj.d(:,j) = mean(tmp_d(:,:,j),2);
            end
            %% Douglas-Peucker for position
            if any(obj.c == 2)
                tmp_ps = zeros(3,M);
                tmp_qs = repmat([1 0 0 0]',[1,M]);
                tmp_counter = 0;
                tmp_cID = (1:obj.K); tmp_cID = tmp_cID(tmp_cID == 2);
                for i = 1:M
                    tmpP = DemoData(i).p(:, DemoData(i).class == 2);
                    tmpQ = DemoData(i).q(:, DemoData(i).class == 2);
                    [~,~,tmpID] = iceDouglasPeuckerItera(tmpP, obj.param_thdDP);
                    if ~isempty(tmpID)
                        tmp_counter = tmp_counter + 1;
                        tmp_ps(:,i) = tmpP(:, tmpID);
                        tmp_qs(:,i) = tmpQ(:, tmpID);
                    end
                end
                if tmp_counter > M/2
                    obj.K = obj.K + 1;
                    obj.c = obj.vecInsert(obj.c, 2, tmp_cID);
                    obj.p = obj.vecInsert(obj.p, mean(tmp_ps,2), tmp_cID);
                    obj.q = obj.vecInsert(obj.q, quatAverage(tmp_qs), tmp_cID);
                    obj.d = obj.vecInsert(obj.d,  vNormalize(obj.p(:,tmp_cID) - obj.p(:,tmp_cID-1)), tmp_cID);
                    obj.d(:, tmp_cID+1) = vNormalize(obj.p(:, tmp_cID+1) - obj.p(:, tmp_cID));
                end
            end
            %% Learn the alignment direction
            if obj.K > 1
                obj.d(:,1) = Rl * obj.d(:,2);
            end
            %% Correction
            obj.p(:,1) = obj.p(:,1) + obj.d(:,1) * obj.param_corr0;
            obj.p(:,1) = obj.p(:,1) + [obj.param_corrx, obj.param_corry, 0]';
            %% Learn the pre-alignment position
            obj.p0 = obj.p(:,1) - obj.param_l * obj.d(:,1);
        end
        [assTraj, DemosA3PDTW] = genAssTraj_dualPolicy(obj, DemoData, N, WinDTW, Ks, hs);
    end
    
    methods (Static = true, Access = public)
        function [DemoData] = constructDemoData(M)
            %constructDemoData Construct the DemoData struct
            %   M: Integer >0, the number of structs. (Default: 1)
            %   -----------------------------------------
            %   DemoData: 1 x M structs,
            %   |   N: Integer >0, the number of data.
            %   |   p: 3 x N, the position data.
            %   |   q: 3 x N, the orientation data.
            %   |   class: 1 x N, the classifications.
            %   @Greengrape5S1_MoCap
            if nargin < 1
                M = 1;
            end
            DemoData = [];
            DemoData.N = 0;
            DemoData.p = zeros(3,1);
            DemoData.q = [1 0 0 0]';
            DemoData.class = 0;
            if M > 1
                DemoData = repmat(DemoData, [1,M]);
            end
        end
        function [trajP] = genSphereP(trajQ, p0)
            %genSphereP Generate sphere positions.
            %   trajQ: 4 x N, the unit quaternion
            %   p0: 3 x 1, the stationary point (Default: [1 1 1]')
            %   -----------------------------------------
            %   trajP: 3 x N, the sphere positions.
            %   @Greengrape5S1_MoCap
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
        function [vectorsOut] = vecInsert(vectorsIn, vec, ID)
            %vecInsert Insert a new vector into the vector series.
            %   vectorsIn: D x K, the vector series.
            %   vec: D x 1, the vector to insert.
            %   ID: Integer >0, the ID of the inserted vector.
            %   -----------------------------------------
            %   vector3Out: D x K+1, the new vector series.
            %   @Greengrape5S1_MoCap
            vectorsOut = [vectorsIn, vec];
            vectorsOut(:,ID+1:end) = vectorsIn(:,ID:end);
            vectorsOut(:,ID) = vec;
        end
        function [p1, q1] = expTrans(pa,qa, p, q)
            %expTrans Transform the data to its experiment coordinates.
            %   pa: 3 x 1, the final position.
            %   qa: 4 x 1, the final orientation.
            %   p: 3 x N, the position data.
            %   q: 4 x N, the orientation data. (Optional)
            %   -----------------------------------------
            %   p1: 3 x N, the transformed position data.
            %   q1: 4 x N, the transformed orientation data.
            Ha = pq2SE3(pa,qa);
            if nargin < 4
                q = repmat([1 0 0 0]', [1, size(p,2)]);
            end
            H = pq2SE3(p,q);
            H1 = H;
            for i = 1:size(H,3)
                H1(:,:,i) = Ha * H(:,:,i);
            end
            p1 = permute(H1(1:3,4,:),[1,3,2]);
            q1 = tform2quat(H1)';
        end
    end
end

