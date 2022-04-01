clear;clear memory;
addpath('./tools')
dataname = 'mvCifar10';
nbits_set = [16];
%[8,16,32,48,64,96,128];
%result number 10
%% Load dataset
load('mvCifar10.mat')
for it = 1:3
    Dis = EuDist2(X{it},Anchor{it},0);
    sigma = mean(mean(Dis)).^0.5;
    feavec = exp(-Dis/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feavec', mean(feavec',2));
end

view_num = size(X,2);
data_our.gnd = gnd+1;
gnd = gnd+1;
% Separate Train and Test Index
for n_iters = 1:1
tt_idx = [];
for ind = 1:10
    list = find(ind==gnd);
    tt_idx = [tt_idx; randsample(list , 100)];
end
list = 1:numel(gnd);
list(tt_idx) = [];
tr_idx = list; 
ttgnd = gnd(tt_idx);
trgnd = gnd(tr_idx);
% clear gnd;

for ii=1:length(nbits_set)
    nbits = nbits_set(ii);
    data_our.indexTrain= tr_idx;
    data_our.indexTest= tt_idx;
    ttfea = cell(1,view_num);
    for view = 1:view_num
        data_our.X{view} = normEqualVariance(X{view}')';
        ttfea{view} = data_our.X{view}(:,tt_idx);
    end

        pars.beta       = 10; % parameters\lambda.5
        pars.gamma    = 0.01; % parameters\gamma1000
        pars.lambda = 0.1;% parameters\eta.1
%         pars.theta  = .1;
        pars.Iter_num = 4;
        pars.nbits    = nbits;
        pars.r = 3;
        
        [B_trn,U1,U2,U3, W, U_W, R, alpha, trtime] = DFMH_fun(data_our,pars);
        
        % for testing
        H = zeros(nbits,length(ttgnd));
        for ind = 1:size(ttfea,2)
            H = H+alpha(ind)*U3{ind}'*U2{ind}'*U1{ind}'*ttfea{ind};
        end
        B_tst = H'*U_W >0;
        
        % B_tst = ttfea{2}'*W_bar{2}>0;
        
        WtrueTestTraining = bsxfun(@eq, ttgnd, trgnd');

        %% Evaluation
        B1 = compactbit(B_trn);
        B2 = compactbit(B_tst);
        DHamm = hammingDist(B2, B1);
        [~, orderH] = sort(DHamm, 2);
        MAP = calcMAP(orderH, WtrueTestTraining);
        fprintf('iter = %d, Bits: %d, MAP: %.4f...   \n', n_iters, nbits, MAP);

%         hammRadius = 2;
%         % hash lookup: precision and reall
%         Ret = (DHamm <= hammRadius+0.00001);
%         [PreR2, Rec] = evaluate_macro(WtrueTestTraining, Ret);
%         fprintf('Iters = %d, Bits: %d, Pre@Radius2: %.4f...   \n', n_iters, nbits, PreR2);
% 
%         topK_MAP = 5000;
%         topKmap = fastMAP(WtrueTestTraining, orderH', topK_MAP);
%         fprintf('Iters = %d, Bits: %d, topK@5000: %.4f...   \n',n_iters, nbits, topKmap);
% 
%         Y = sparse(1:length(trgnd), double(trgnd),1); Y = full(Y);
%         Ytt = sparse(1:length(ttgnd), double(ttgnd),1); Ytt = full(Ytt);
% 
%         top_NDCG = 100;
%         ndcg = NDCG_k(Y', Ytt',orderH',top_NDCG);
%         fprintf('Iters = %d, Bits: %d, NDCG_100: %.4f...   \n',n_iters, nbits, ndcg);

        pos = [1:10:40 50:50:1000];
        [rec_pos, pres_pos] = recall_precision5(WtrueTestTraining, DHamm, pos);

        [recall, precision, ~] = recall_precision(WtrueTestTraining, DHamm);
        [MAP1] = area_RP(recall, precision);
%         fprintf('Iters = %d, Bits: %d, MAP1: %.4f...   \n',n_iters, nbits, MAP1);
        
        name = ['./Results/DFMH_'  dataname '_nbit_' num2str(nbits) '_' num2str(n_iters)];
%         save(name,'MAP','MAP1','PreR2','recall', 'precision','rec_pos','pres_pos','topKmap','ndcg','WtrueTestTraining','B1','B2','trtime')
        save(name,'MAP','MAP1','recall', 'precision','rec_pos','pres_pos','WtrueTestTraining','B1','B2','trtime')
       
%         % Store all data
%         mapMat(n_iters) = MAP;
%         map1Mat(n_iters) = MAP1;
%         PreR2Mat(n_iters) = PreR2;
%         topKmapMat(n_iters) = topKmap;
%         ndcgMat(n_iters) = ndcg;
%         recposMat(n_iters,:) = rec_pos;
%         presposMat(n_iters,:) = pres_pos;
%         recallMat(n_iters,:) = recall;
%         precisionMat(n_iters,:) = precision;

        clear recall precision MAP rec_pos pres_pos
end
    
% Store all data

% [~,indx] = max(mapMat);
% MAP = mapMat(indx);
% MAP1 = map1Mat(indx);
% PreR2 = PreR2Mat(indx);
% topKmap = topKmapMat(indx);
% ndcg = ndcgMat(indx);
% rec_pos = recposMat(indx,:);
% pres_pos = presposMat(indx,:);
% recall = recallMat(indx,:);
% precision = precisionMat(indx,:);
% clear mapMat map1Mat PreR2Mat topKmapMat ndcgMat recposMat presposMat recallMat precisionMat
% 
% name = ['./Results/ours_'  dataname '_nbit_' num2str(nbits) '_' num2str(myiter)];
% save(name,'MAP','MAP1','PreR2','recall', 'precision','rec_pos','pres_pos','topKmap','ndcg','WtrueTestTraining','B1','B2','trtime')
% 
end