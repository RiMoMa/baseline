function histogramas ()

%%%ruta de dataset
conf.calDir = 'data_test/' ;

carpeta = dir(conf.calDir) ;
carpeta = carpeta([carpeta.isdir]) ;
carpeta = {carpeta(3:end).name} ;


for i=1:length(carpeta)
        ruta=dir(fullfile(conf.calDir, carpeta{i} ,'/frames/x40/' ,'*tiff'))';
     
       
       
       
        for n=1:length(ruta)
            
     im = imread(fullfile(conf.calDir,cell2mat(carpeta(i)), '/frames/x40/',ruta(n).name)) ;
       aux=im;
       
       
       %%%%%%%%%%%%%%%deteccion de canditos%%%%%%%%%%%%%5
   %separación H&E
    [Inorm1 H1 E1] = normalizeStaining(aux);
    rgb_aux=rgb2gray(H1);
    rgb_aux=medfilt2(rgb_aux);
    I=uint8(rgb_aux);
    I=impyramid(I,'reduce');
    I=impyramid(I,'expand');
    rgb_aux=I;
     %%%%%%% MSER %%%%%%% 
     
     disp('Calculando MSER features...');
     [r,f]=vl_mser(rgb_aux,'Delta',5,'DarkOnBright',1,'BrightOndark',0,'MaxArea',0.0010, 'MinDiversity',0.8,'MinArea',0.0001);
    %%corte de los candidatos
     
       f=vl_ertr(f);
       nCandidatos=size(r,1);
    fprintf('Numero total de candidatos %d \n', nCandidatos);
    im=rgb_aux;
    d={};
   aux_sel=randperm(length(f));
  d=ones(size(I));
       nCandidatos=size(r,1);
   M = zeros(size(I));
    Mm = zeros(size(I));
   for x=r'        
    s = vl_erfill(I,x) ;
    M(s) = M(s) + 1;
   end
   
   
    %Operaciones morfologicas para eliminar huecos y separa nucleos
	xM=M;
	M=and(M,d);
 
	M=imfill(M,'holes');
    se=strel('square',4);
    M=imerode(M,se);
    M=imopen(M,se);
   if length(aux_sel)<1600
       palabras=length(aux_sel);
   else
       palabras=1600;
   end
   
      vector1 = f(1,[aux_sel]);
       vector2 = f(2,[aux_sel]);
	fc=[vector1;vector2];
    
    [area_pixels,perimetro_pixels,factor_forma,RC]=medidasroundness (M,0.2468,50,fc);
    f_mrdescr=fc;
    c=[area_pixels;perimetro_pixels;factor_forma;RC];
    destino=cell2mat(carpeta(i));
    cd histogramas/;
    mkdir (destino);
    cd ..
    aux_name=['histogramas/',destino,'/',ruta(n).name,'.mat'];
    minimos=[min(area_pixels),min(perimetro_pixels),min(factor_forma),min(RC)];
    maximos=[max(area_pixels),max(perimetro_pixels),max(factor_forma),max(RC)];

     save(aux_name,'area_pixels','perimetro_pixels','factor_forma','RC','minimos','maximos') ;      
     
        end

    
    
end
