clc;clf;close all;clear all;


load 'data_151116_zr01_s' %
t_end=tout(end);
    

   trnum=5;   %%%CSPのトレーニングデータ
  tesnum=6;   %%%CSPのテストデータ


C=[ones(1,1024),100*ones(1,1024)];
   


%CSPトレーニング%%
    
  %%%データの読み込み(1周期)%%% 
    
    
 
    
    Ncycle = trnum; % 0~7
    
   
    
    Nstart=Ncycle*16*128+1; Ni=1; Nf=Nstart+128*16-1;
    
    
    
     x1=yout(Nstart:Ni:Nf,20:23);
    
        
        X1=x1';
   
   
    
    %%CSPの計算%%%
    class1=X1(:,1:1024);
    class2=X1(:,1025:2048);

    [PTranspose] = CSP(class1,class2);

    classtrain= horzcat(class1,class2);
    
    train = spatFilt(classtrain,PTranspose,2);
    
    %%プロット%%%

    subplot(2,2,1)
   plot(tout(Nstart:Ni:Nf,1),yout(Nstart:Ni:Nf,20:23)); grid;
   title('PSD[9 12 15 18]')
   
   
    subplot(2,2,2)
    
    scatter(train(1,:),train(2,:),10,C);grid
   %scatter3(train(1,:),train(2,:),train(3,:),5,C);
   title('CSP train result')

    
    
    
    % linear bayes classifier
    
     %%%データの読み込み(1周期)%%% 
    
    
 
    
    Ncycle = tesnum; % 0~7
    Nstart=Ncycle*16*128+1; Ni=1; Nf=Nstart+128*16-1;
    
    x2=yout(Nstart:Ni:Nf,20:23);
    
        
        X2=x2'; 
   
  
    
    
    class1=X2(:,1:1024);
    class2=X2(:,1025:2048);

    classtest= horzcat(class1,class2);
    
    test = spatFilt(classtest,PTranspose,3);
    
    %%プロット%%%
  
 subplot(2,2,3)      
   plot(tout(Nstart:Ni:Nf,1),yout(Nstart:Ni:Nf,20:23)); grid;
   title('PSD[9 12 15 18]')

    
    subplot(2,2,4)
    scatter(test(1,:),test(2,:),10,C);grid
   % scatter3(test(1,:),test(2,:),test(3,:),5,C);
    title('CSP test result')
    
  
 