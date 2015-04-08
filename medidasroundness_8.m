%Parametros de corte
%resolucion del microscopio en pixels um/pixels

function [area_pixels,perimetro_pixels,factor_forma,RC]= medidasroundness (M,resolucion,areamedianucleo,f)


%resolucion=0.2455;
%areamedianucleo=50;
%areamaxima=;
pixels= areamedianucleo/resolucion*0.95;
pixels=uint8(pixels);

%%%operacion apertura
se=strel('disk',5);
M=imopen(M,se);

M_aux=bwareaopen(M,double(pixels));
%M_aux=M;
%cortar

sz=size(f,2); %numero candidatos
parche=cell(sz,1);
distancia=64; %tamaño del parche

   [L,N]=bwlabel(M_aux);
   
   %%%%% calculo de area y area del perimetro
   im_area=bwconncomp(M_aux);
   area_pixels=cellfun(@numel,im_area.PixelIdxList);
   im_perimetro=bwconncomp(bwperim(L));

   perimetro_pixels=cellfun(@numel,im_perimetro.PixelIdxList);
   
for x=1:length(area_pixels)
    x_min=f(1,x);
    y_min=f(2,x);

    factor_forma(x)=perimetro_pixels(x)^2/area_pixels(x);
    %Razon de circularidad RC=4*pi/factor_forma
    RC(x)=4*pi/(factor_forma(x));
    
end

end



