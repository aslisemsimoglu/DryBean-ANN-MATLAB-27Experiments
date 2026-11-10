%% YSA-3 L2 analysis: publication-ready visuals (MATLAB R2026b)
clearvars; clc; close all;
root = fullfile(getenv('USERPROFILE'),'Documents','YSA_Vize_DryBean');
inDir = fullfile(root,'11_L2_Ek_Deneyler');
outDir = fullfile(inDir,'Gorseller');
assert(isfolder(inDir),'11_L2_Ek_Deneyler klasoru bulunamadi.');
if ~isfolder(outDir), mkdir(outDir); end
D = readtable(fullfile(inDir,'L2_Karsilastirma.csv'),'TextType','string');
S = readtable(fullfile(inDir,'L2_Grup_Ozeti.csv'),'TextType','string');
assert(height(D)==12 && height(S)==4,'Beklenen 12 kosu / 4 grup bulunamadi.');
D = sortrows(D,{'L2','Seed'}); S = sortrows(S,'L2');
labels = {'0','0.0001','0.001','0.01'};
x = 1:4;
clr = [0.12 0.35 0.58;0.38 0.50 0.63;0.12 0.58 0.48;0.75 0.40 0.22];
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial');
%% Figure 1: mean accuracy and sample SD
f=figure('Color','w','Position',[80 80 1100 650]);
h=bar(x,S.mean_TestAccuracy,0.60,'FaceColor','flat','EdgeColor','none');
h.CData=clr; hold on;
errorbar(x,S.mean_TestAccuracy,S.std_TestAccuracy,'k.','LineWidth',1.5,'CapSize',13);
for k=1:4
 text(x(k),S.mean_TestAccuracy(k)+S.std_TestAccuracy(k)+0.025, ...
     sprintf('%.3f%%',S.mean_TestAccuracy(k)),'HorizontalAlignment','center','FontWeight','bold');
end
xticks(x); xticklabels(labels); xlabel('L2 duzenlilestirme katsayisi');
ylabel('Ortalama test dogrulugu (%)');
title('YSA-3: L2 Katsayisina Gore Test Dogrulugu (3 Seed)');
subtitle('Hata cubuklari: ornek standart sapma (yuzde puan)');
ylim([92.25 93.55]); grid on; box off;
exportgraphics(f,fullfile(outDir,'Sekil_L2_Test_Dogrulugu.png'),'Resolution',300);
exportgraphics(f,fullfile(outDir,'Sekil_L2_Test_Dogrulugu.pdf'),'ContentType','vector'); close(f);
%% Figure 2: paired seed profiles
f=figure('Color','w','Position',[80 80 1100 650]); hold on;
seeds=unique(D.Seed)'; marks={'o','s','^'};
for j=1:numel(seeds)
 rows=D(D.Seed==seeds(j),:); rows=sortrows(rows,'L2');
 plot(x,rows.TestAccuracy,'-','Marker',marks{j},'LineWidth',2.2, ...
     'MarkerSize',8,'DisplayName',sprintf('Seed %d',seeds(j)));
end
xticks(x);xticklabels(labels);xlabel('L2 duzenlilestirme katsayisi');
ylabel('Test dogrulugu (%)');title('Eslesmis Tohumlarda Test Dogrulugu');
legend('Location','southwest');grid on;box off;
exportgraphics(f,fullfile(outDir,'Sekil_L2_Seed_Bazli.png'),'Resolution',300);
exportgraphics(f,fullfile(outDir,'Sekil_L2_Seed_Bazli.pdf'),'ContentType','vector');close(f);
%% Figure 3: minimum validation loss
f=figure('Color','w','Position',[80 80 1100 650]);
h=bar(x,S.mean_MinValLoss,0.6,'FaceColor','flat','EdgeColor','none');h.CData=clr;hold on;
errorbar(x,S.mean_MinValLoss,S.std_MinValLoss,'k.','LineWidth',1.5,'CapSize',13);
xticks(x);xticklabels(labels);xlabel('L2 duzenlilestirme katsayisi');
ylabel('Minimum dogrulama kaybi (ortalama)');
title('L2 Katsayisina Gore Minimum Dogrulama Kaybi');
subtitle('Hata cubuklari: 3 seed uzerinden ornek standart sapma');
grid on;box off;
exportgraphics(f,fullfile(outDir,'Sekil_L2_Min_Dogrulama_Kaybi.png'),'Resolution',300);
exportgraphics(f,fullfile(outDir,'Sekil_L2_Min_Dogrulama_Kaybi.pdf'),'ContentType','vector');close(f);
%% Figure 4: training / validation / test means
f=figure('Color','w','Position',[80 80 1100 650]);
Y=[S.mean_TrainAccuracy S.mean_ValAccuracy S.mean_TestAccuracy];
bar(x,Y,'grouped');xticks(x);xticklabels(labels);
xlabel('L2 duzenlilestirme katsayisi');ylabel('Ortalama dogruluk (%)');
title('Egitim, Dogrulama ve Test Basariminin Karsilastirilmasi');
legend({'Egitim','Dogrulama','Test'},'Location','southoutside','Orientation','horizontal');
grid on;box off;
exportgraphics(f,fullfile(outDir,'Sekil_L2_Train_Val_Test.png'),'Resolution',300);
exportgraphics(f,fullfile(outDir,'Sekil_L2_Train_Val_Test.pdf'),'ContentType','vector');close(f);
%% Table: clean CSV and text summary
R=table(S.L2,S.mean_TrainAccuracy,S.mean_ValAccuracy,S.mean_TestAccuracy, ...
 S.std_TestAccuracy,S.mean_TestMacroF1,S.mean_MinValLoss, ...
 'VariableNames',{'L2','TrainOrt','ValOrt','TestOrt','TestStd','MacroF1Ort','MinValLossOrt'});
writetable(R,fullfile(outDir,'Tablo_L2_Ozet.csv'));
base=S(S.L2==0.0001,:);
R.DeltaTestVsOriginal_pp=R.TestOrt-base.mean_TestAccuracy;
writetable(R,fullfile(outDir,'Tablo_L2_Ozet_Delta.csv'));
fprintf('\n=== PROFESYONEL GORSELLER HAZIR ===\n%s\n',outDir);
disp(R);
fprintf('NOT: Test sonuclari raporlama icindir; L2 secimini test sonucuna gore yapmayin.\n');
