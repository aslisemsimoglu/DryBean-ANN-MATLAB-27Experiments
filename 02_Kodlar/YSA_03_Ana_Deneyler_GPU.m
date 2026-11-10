%% YSA VIZE - 27 KOSULUK OTOMATIK DENEY SISTEMI (MATLAB R2026b)
% Dry Bean Dataset | 3 mimari x 3 oran x 3 tekrar
% Her kosu egitimden hemen sonra kaydedilir; yeniden calistirmada devam eder.
% Ilk CPU modeli varsa korunur; kalan kosular GPU'da egitilir.
clearvars; clc; close all;
proje = fullfile(getenv('USERPROFILE'),'Documents','YSA_Vize_DryBean');
veriDosyasi = fullfile(proje,'01_Veri','DryBean_VeriBolmeleri.mat');
assert(isfile(veriDosyasi),'Veri bolmeleri bulunamadi: %s',veriDosyasi);
S = load(veriDosyasi,'veriBolmeleri'); veriBolmeleri = S.veriBolmeleri;
assert(numel(veriBolmeleri)==3,'Uc veri bolmesi bekleniyor.');
klasorModel = fullfile(proje,'03_Modeller');
klasorGrafik = fullfile(proje,'04_Egitim_Grafikleri');
klasorCM = fullfile(proje,'05_Karisiklik_Matrisleri');
klasorTablo = fullfile(proje,'06_Sonuc_Tablolari');
klasorLog = fullfile(proje,'10_Loglar');
for p={klasorModel,klasorGrafik,klasorCM,klasorTablo,klasorLog}
    if ~isfolder(p{1}), mkdir(p{1}); end
end
logDosyasi=fullfile(klasorLog,'Ana_Deneyler_GPU_Gunluk.txt');
diary(logDosyasi); diary on;
cleanupDiary=onCleanup(@() diary('off'));

% Ilk denemede 1, sonrasinda 27 yapabilirsiniz.
MAX_YENI_EGITIM = 25;
mimariler = {[10],[20 10],[32 16 8]};
tohumlar = [101 202 303];
oranKodlari = {'60_40','70_30','80_20'};
kolonlar={'DeneyID','Mimari','Oran','Tekrar','Tohum', ...
    'TrainN','ValN','TestN','TrainAccuracy','ValAccuracy', ...
    'TestAccuracy','TestMacroF1','TestCrossEntropy','SureSaniye','Ortam'};
sonucDosyasi=fullfile(klasorTablo,'Tum_Deney_Sonuclari.csv');

fprintf('\n=== YSA ANA DENEYLER: GPU / KAYITTAN DEVAM ===\n');
yeniEgitim=0;
for oran=1:3
    D=veriBolmeleri(oran);
    XTrain=single(D.XTrain); XVal=single(D.XVal); XTest=single(D.XTest);
    YTrain=D.YTrain; YVal=D.YVal; YTest=D.YTest;
    siniflar=categories(YTrain); K=numel(siniflar);
    assert(size(XTrain,2)==16 && K==7,'Veri boyutu/sinif sayisi beklenmiyor.');
    for mimari=1:3
        for tekrar=1:3
            kimlik=sprintf('YSA%02d_Oran%s_Tekrar%02d',mimari,oranKodlari{oran},tekrar);
            modelDosyasi=fullfile(klasorModel,[kimlik '.mat']);
            if ~isfile(modelDosyasi)
                if yeniEgitim >= MAX_YENI_EGITIM
                    fprintf('\nBu oturumdaki yeni egitim limiti doldu (%d).\n',MAX_YENI_EGITIM);
                    YSA_tabloyu_yenile(klasorModel,sonucDosyasi,kolonlar,veriBolmeleri,oranKodlari,tohumlar);
                    return
                end
                fprintf('\nEGITIM BASLADI: %s\n',kimlik);
                rng(tohumlar(tekrar),'twister');
                layers=[featureInputLayer(16,Normalization='none',Name='input')];
                for h=1:numel(mimariler{mimari})
                    layers=[layers; fullyConnectedLayer(mimariler{mimari}(h), ...
                        Name=sprintf('hidden%d',h)); reluLayer(Name=sprintf('relu%d',h))]; %#ok<AGROW>
                end
                layers=[layers; fullyConnectedLayer(K,Name='output'); softmaxLayer(Name='softmax')];
                opts=trainingOptions('adam',MaxEpochs=50,MiniBatchSize=128, ...
                    InitialLearnRate=0.001,Shuffle='every-epoch', ...
                    ValidationData={XVal,YVal},ValidationFrequency=20, ...
                    Metrics='accuracy',Plots='none',Verbose=false, ...
                    ExecutionEnvironment='gpu');
                tic;
                [net,info]=trainnet(XTrain,YTrain,layers,'crossentropy',opts);
                sure=toc;
                ortam='GPU';
                % ONCE kaydet: sonraki hesaplamalarda hata olsa da egitim kaybolmaz.
                save(modelDosyasi,'net','info','layers','opts','kimlik', ...
                    'mimari','oran','tekrar','siniflar','sure','ortam','-v7.3');
                yeniEgitim=yeniEgitim+1;
                fprintf('EGITIM KAYDEDILDI: %s (%.1f sn)\n',kimlik,sure);
            else
                fprintf('KAYITLI MODEL: %s\n',kimlik);
            end
            % Mevcut modelden eksik metrikleri/grafikleri tamamla.
            try
                YSA_kosuyu_tamamla(modelDosyasi,D,klasorGrafik,klasorCM,klasorTablo,kimlik);
            catch ME
                warning('Sonuc isleme hatasi (%s): %s',kimlik,ME.message);
                fprintf('Model korunuyor; sonraki calistirmada tekrar denenecek.\n');
            end
            YSA_tabloyu_yenile(klasorModel,sonucDosyasi,kolonlar,veriBolmeleri,oranKodlari,tohumlar);
        end
    end
end
fprintf('\n=== ISLEM BITTI | MODEL SAYISI: %d / 27 ===\n', ...
    numel(dir(fullfile(klasorModel,'YSA*_Oran*_Tekrar*.mat'))));
fprintf('Sonuc tablosu: %s\n',sonucDosyasi);

function YSA_kosuyu_tamamla(modelDosyasi,D,klasorGrafik,klasorCM,klasorTablo,kimlik)
    A=load(modelDosyasi);
    metrikDosyasi=fullfile(klasorTablo,[kimlik '_Metrikler.mat']);
    if ~isfile(metrikDosyasi)
        siniflar=categories(D.YTrain);
        sTrain=double(gather(minibatchpredict(A.net,D.XTrain)));
        sVal=double(gather(minibatchpredict(A.net,D.XVal)));
        sTest=double(gather(minibatchpredict(A.net,D.XTest)));
        pTrain=scores2label(sTrain,siniflar);
        pVal=scores2label(sVal,siniflar);
        pTest=scores2label(sTest,siniflar);
        accTrain=100*mean(pTrain==D.YTrain);
        accVal=100*mean(pVal==D.YVal);
        accTest=100*mean(pTest==D.YTest);
        order=categorical(siniflar,siniflar);
        C=confusionmat(D.YTest,pTest,'Order',order);
        tp=diag(C);
        precision=tp./max(sum(C,1)',1);
        recall=tp./max(sum(C,2),1);
        f1=2*precision.*recall./max(precision+recall,eps);
        macroF1=mean(f1);
        [~,gercekInd]=ismember(cellstr(D.YTest),siniflar);
        assert(all(gercekInd>0),'Bilinmeyen sinif bulundu.');
        assert(size(sTest,1)==numel(gercekInd),'Tahmin boyutu uyusmuyor.');
        ind=sub2ind(size(sTest),(1:numel(gercekInd))',gercekInd);
        testCE=-mean(log(max(sTest(ind),1e-12)));
        save(metrikDosyasi,'accTrain','accVal','accTest','macroF1', ...
            'testCE','C','pTest','precision','recall','f1','-v7.3');
        sinifTablosu=table(string(siniflar),precision,recall,f1, ...
            'VariableNames',{'Sinif','Precision','Recall','F1'});
        writetable(sinifTablosu,fullfile(klasorTablo,[kimlik '_Sinif_Metrikleri.csv']));
        fprintf('METRIK: %s | Train %.2f | Val %.2f | Test %.2f | F1 %.4f\n', ...
            kimlik,accTrain,accVal,accTest,macroF1);
    end
    E=A.info.TrainingHistory; V=A.info.ValidationHistory;
    fAcc=fullfile(klasorGrafik,[kimlik '_Accuracy.png']);
    if ~isfile(fAcc)
        fig=figure('Visible','off','Color','white');
        plot(E.Iteration,E.Accuracy,'LineWidth',1.3); hold on;
        plot(V.Iteration,V.Accuracy,'LineWidth',1.8);
        xlabel('Iteration'); ylabel('Accuracy (%)');
        title(strrep([kimlik ' | Accuracy'],'_',' '));
        legend('Training','Validation','Location','best'); grid on;
        exportgraphics(fig,fAcc,'Resolution',200); close(fig);
    end
    fLoss=fullfile(klasorGrafik,[kimlik '_Loss.png']);
    if ~isfile(fLoss)
        fig=figure('Visible','off','Color','white');
        plot(E.Iteration,E.Loss,'LineWidth',1.3); hold on;
        plot(V.Iteration,V.Loss,'LineWidth',1.8);
        xlabel('Iteration'); ylabel('Cross-Entropy Loss');
        title(strrep([kimlik ' | Loss'],'_',' '));
        legend('Training','Validation','Location','best'); grid on;
        exportgraphics(fig,fLoss,'Resolution',200); close(fig);
    end
    fCM=fullfile(klasorCM,[kimlik '_ConfusionMatrix.png']);
    if ~isfile(fCM)
        M=load(metrikDosyasi,'pTest');
        fig=figure('Visible','off','Color','white','Position',[100 100 950 750]);
        chart=confusionchart(fig,D.YTest,M.pTest);
        chart.Title=strrep([kimlik ' | Test Confusion Matrix'],'_',' ');
        chart.RowSummary='row-normalized';
        chart.ColumnSummary='column-normalized';
        exportgraphics(fig,fCM,'Resolution',200); close(fig);
    end
end

function YSA_tabloyu_yenile(klasorModel,sonucDosyasi,kolonlar,bolmeler,oranKodlari,tohumlar)
    L=dir(fullfile(klasorModel,'YSA*_Oran*_Tekrar*.mat'));
    rows=cell(0,numel(kolonlar));
    for q=1:numel(L)
        path=fullfile(L(q).folder,L(q).name);
        A=load(path,'kimlik','mimari','oran','tekrar','sure','ortam');
        if ~isfield(A,'kimlik'), continue; end
        % Ilk CPU modeli sadece kimlik ile kaydedilmisti.
        tok=regexp(char(A.kimlik),'^YSA(\d+)_Oran(60_40|70_30|80_20)_Tekrar(\d+)$','tokens','once');
        if isempty(tok), continue; end
        if ~isfield(A,'mimari'), A.mimari=str2double(tok{1}); end
        if ~isfield(A,'oran'), A.oran=find(strcmp(oranKodlari,tok{2}),1); end
        if ~isfield(A,'tekrar'), A.tekrar=str2double(tok{3}); end
        mPath=fullfile(fileparts(sonucDosyasi),[A.kimlik '_Metrikler.mat']);
        if ~isfile(mPath), continue; end
        M=load(mPath);
        D=bolmeler(A.oran);
        if ~isfield(A,'sure'), A.sure=NaN; end
        if ~isfield(A,'ortam'), A.ortam='CPU'; end
        rows(end+1,:)={string(A.kimlik),A.mimari, ...
            string(oranKodlari{A.oran}),A.tekrar,tohumlar(A.tekrar), ...
            numel(D.YTrain),numel(D.YVal),numel(D.YTest), ...
            M.accTrain,M.accVal,M.accTest,M.macroF1,M.testCE, ...
            A.sure,string(A.ortam)}; %#ok<AGROW>
    end
    if isempty(rows)
        fprintf('Henuz tam metrikli deney yok.\n'); return
    end
    sonuc=cell2table(rows,'VariableNames',kolonlar);
    sonuc=sortrows(sonuc,{'Oran','Mimari','Tekrar'});
    writetable(sonuc,sonucDosyasi);
    fprintf('CSV GUNCELLENDI: %d / 27 kosu\n',height(sonuc));
end
