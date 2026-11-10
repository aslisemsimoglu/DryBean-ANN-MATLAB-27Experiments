%% YSA VIZE - L2 DUZENLILESTIRME EK DENEYLERI (MATLAB R2026b)
% Ana 27 kosuyu degistirmez. 3 seed x 3 L2 = 9 yeni egitim.
clearvars; clc; close all;
proje=fullfile(getenv('USERPROFILE'),'Documents','YSA_Vize_DryBean');
veriDosyasi=fullfile(proje,'01_Veri','DryBean_VeriBolmeleri.mat');
assert(isfile(veriDosyasi),'Veri bolmeleri bulunamadi.');
S=load(veriDosyasi,'veriBolmeleri'); D=S.veriBolmeleri(1);
XTrain=single(D.XTrain); XVal=single(D.XVal); XTest=single(D.XTest);
YTrain=D.YTrain; YVal=D.YVal; YTest=D.YTest;
siniflar=categories(YTrain); K=numel(siniflar);
assert(size(XTrain,2)==16 && K==7,'Beklenmeyen veri boyutu.');
assert(size(XTrain,1)==7351 && size(XVal,1)==816 && size(XTest,1)==5444,'Beklenmeyen veri bolmesi.');

kok=fullfile(proje,'11_L2_Ek_Deneyler');
modelDir=fullfile(kok,'Modeller'); grafikDir=fullfile(kok,'Grafikler');
for p={kok,modelDir,grafikDir}, if ~isfolder(p{1}), mkdir(p{1}); end, end
csvPath=fullfile(kok,'L2_Karsilastirma.csv');
seeds=[101 202 303]; lambdas=[0 0.001 0.01];
rows=cell(0,12);
fprintf('\n=== L2 EK DENEYLERI: YSA-3 / 60-40 ===\n');
for s=1:numel(seeds)
    seed=seeds(s);
    % Mevcut kontrol modeli ve metrikleri (L2=0 oldugu koddan dogrulanmali)
    id=sprintf('YSA03_Oran60_40_Tekrar%02d',s);
    baselineModel=fullfile(proje,'03_Modeller',[id '.mat']);
    baselineMetrics=fullfile(proje,'06_Sonuc_Tablolari',[id '_Metrikler.mat']);
    assert(isfile(baselineModel) && isfile(baselineMetrics), ...
        'Kontrol dosyasi eksik: %s',id);
    B=load(baselineModel,'info','opts'); M=load(baselineMetrics);
    assert(isfield(B,'opts'),'Kontrol modelinde opts eksik.');
    baselineL2=B.opts.L2Regularization;
    fprintf('Orijinal L2 katsayisi (seed %d): %.8g\n',seed,baselineL2);
    assert(~ismembertol(baselineL2,lambdas,1e-12), ...
        'Orijinal L2 yeni deneylerden biriyle ayni; deneyi durdurup tasarimi kontrol edin.');
    [bMin,bEpoch,bLast]=validationStats(B.info);
    rows(end+1,:)={string(id),seed,baselineL2,M.accTrain,M.accVal,M.accTest, ...
        M.macroF1,M.testCE,bMin,bEpoch,bLast,string('original')}; %#ok<AGROW>

    for j=1:numel(lambdas)
        lambda=lambdas(j);
        suffix=sprintf('L2_%g_Seed%d',lambda,seed);
        modelPath=fullfile(modelDir,[suffix '.mat']);
        if isfile(modelPath)
            fprintf('Kayitli deney kullaniliyor: %s\n',suffix);
            A=load(modelPath,'net','info','opts');
        else
            fprintf('Egitim basliyor: %s\n',suffix);
            rng(seed,'twister');
            layers=[featureInputLayer(16,Normalization='none',Name='input'); ...
                fullyConnectedLayer(32,Name='hidden1'); reluLayer(Name='relu1'); ...
                fullyConnectedLayer(16,Name='hidden2'); reluLayer(Name='relu2'); ...
                fullyConnectedLayer(8,Name='hidden3'); reluLayer(Name='relu3'); ...
                fullyConnectedLayer(K,Name='output'); softmaxLayer(Name='softmax')];
            opts=trainingOptions('adam',MaxEpochs=50,MiniBatchSize=128, ...
                InitialLearnRate=0.001,Shuffle='every-epoch', ...
                ValidationData={XVal,YVal},ValidationFrequency=20, ...
                Metrics='accuracy',Plots='none',Verbose=false, ...
                ExecutionEnvironment='gpu',L2Regularization=lambda);
            [net,info]=trainnet(XTrain,YTrain,layers,'crossentropy',opts);
            save(modelPath,'net','info','opts','layers','seed','lambda','-v7.3');
            A=struct('net',net,'info',info,'opts',opts);
        end
        [trainAcc,valAcc,testAcc,macroF1,testCE]=evaluateNet(A.net,XTrain,YTrain,XVal,YVal,XTest,YTest,siniflar);
        [minLoss,minEpoch,lastLoss]=validationStats(A.info);
        rows(end+1,:)={string(suffix),seed,lambda,trainAcc,valAcc,testAcc, ...
            macroF1,testCE,minLoss,minEpoch,lastLoss,string('new')}; %#ok<AGROW>
        % Dogrulama kaybi egirisi: her seed icin ayri PNG
        try
            V=A.info.ValidationHistory;
            f=figure('Visible','off','Color','white');
            plot(V.Iteration,V.Loss,'LineWidth',1.7); grid on;
            xlabel('Iteration'); ylabel('Validation cross-entropy');
            title(sprintf('YSA-3 | seed %d | L2 %.4g',seed,lambda));
            exportgraphics(f,fullfile(grafikDir,[suffix '_ValidationLoss.png']),'Resolution',200);
            close(f);
        catch ME
            warning('Grafik olusturulamadi: %s',ME.message);
        end
        % Her kosudan sonra tabloyu yaz: kesintide kayip yok
        T=cell2table(rows,'VariableNames',{'DeneyID','Seed','L2','TrainAccuracy', ...
            'ValAccuracy','TestAccuracy','TestMacroF1','TestCrossEntropy', ...
            'MinValLoss','MinValEpoch','FinalValLoss','Kaynak'});
        writetable(T,csvPath);
        fprintf('%s | Train %.3f | Val %.3f | Test %.3f | F1 %.4f\n', ...
            suffix,trainAcc,valAcc,testAcc,macroF1);
    end
end
T=cell2table(rows,'VariableNames',{'DeneyID','Seed','L2','TrainAccuracy', ...
    'ValAccuracy','TestAccuracy','TestMacroF1','TestCrossEntropy', ...
    'MinValLoss','MinValEpoch','FinalValLoss','Kaynak'});
writetable(T,csvPath);
summary=groupsummary(T,'L2',{'mean','std'}, ...
    {'TrainAccuracy','ValAccuracy','TestAccuracy','TestMacroF1','MinValLoss'});
writetable(summary,fullfile(kok,'L2_Grup_Ozeti.csv'));
disp(summary);
fprintf('\nTAMAMLANDI: %d kosu (3 kontrol + 9 yeni).\n',height(T));
fprintf('SONUC: %s\n',csvPath);

function [trainAcc,valAcc,testAcc,macroF1,testCE]=evaluateNet(net,XTrain,YTrain,XVal,YVal,XTest,YTest,siniflar)
    tr=double(gather(minibatchpredict(net,XTrain)));
    va=double(gather(minibatchpredict(net,XVal)));
    te=double(gather(minibatchpredict(net,XTest)));
    predTr=scores2label(tr,siniflar);
    predVa=scores2label(va,siniflar);
    predTe=scores2label(te,siniflar);
    trainAcc=100*mean(predTr==YTrain);
    valAcc=100*mean(predVa==YVal);
    testAcc=100*mean(predTe==YTest);
    order=categorical(siniflar,siniflar);
    C=confusionmat(YTest,predTe,'Order',order);
    tp=diag(C);
    precision=tp./max(sum(C,1)',1);
    recall=tp./max(sum(C,2),1);
    f1=2*precision.*recall./max(precision+recall,eps);
    macroF1=mean(f1);
    [~,idx]=ismember(cellstr(YTest),siniflar);
    assert(all(idx>0) && size(te,1)==numel(idx),'Etiket uyusmazligi.');
    testCE=-mean(log(max(te(sub2ind(size(te),(1:numel(idx))',idx)),1e-12)));
end

function [minLoss,minEpoch,lastLoss]=validationStats(info)
    V=info.ValidationHistory;
    good=isfinite(V.Loss);
    assert(any(good),'Dogrulama kaybi bulunamadi.');
    losses=V.Loss(good); epochs=nan(size(V.Loss)); if ismember('Epoch',V.Properties.VariableNames), epochs=V.Epoch; elseif ismember('Iteration',V.Properties.VariableNames), epochs=ceil(V.Iteration/ceil(7351/128)); end; epochs=epochs(good);
    [minLoss,i]=min(losses);
    minEpoch=epochs(i);
    lastLoss=losses(end);
end
