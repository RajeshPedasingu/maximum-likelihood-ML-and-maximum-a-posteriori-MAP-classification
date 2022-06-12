clc;
clear all;
close all;

img_94 = (imread('brain_94.png'));
p11=double(img_94);

img_94_1= imread('label_94_grey.png');
img_94_1=rgb2gray(img_94_1) ;


img_86 = (imread('brain_86.png'));

img_86_1 = (imread('label_86_grey.png'));
img_86_1=rgb2gray(img_86_1) ;

img_99 = (imread('brain_99.png'));


img_99_1 = (imread('label_99_grey.png'));
img_99_1=rgb2gray(img_99_1) ;

img_105 = (imread('brain_105.png'));



img_105_1 = (imread('label_105_grey.png'));
img_105_1=rgb2gray(img_105_1) ;






%class 1
p1=double(img_94_1==85);
n1=sum(sum(img_94_1==85));
m1=sum(sum((p11.*p1)))/n1;

v1=(p11(img_94_1==85));
v1=var(v1(:));

%class 2
p2=double(img_94_1==170);
n2=sum(sum(img_94_1==170));
m2=sum(sum((p11.*p2)))/n2;

v2=(p11(img_94_1==170));
v2=var(v2(:));


% class 3
p3=double(img_94_1==255);
n3=sum(sum(img_94_1==255));
m3=sum(sum((p11.*p3)))/n3;

v3=(p11(img_94_1==255));
v3=var(v3(:));

%prier propabilities
pw1=n1/(numel(img_94));
pw2=n2/(numel(img_94));
pw3=n3/(numel(img_94));







pw=[pw1 pw2 pw3];
m=[m1 m2 m3];
v=[v1 v2 v3];

figure;
imshow(img_94)


title('given image brain_ 94')
n1=ML_classificatino(img_94,img_94_1,m,v,pw);
text="ML_accuracy percentage of img_94: " + num2str(n1);
disp(text)
subplot(1,2,2)
imshow(img_94_1)
title('given lable image')



figure;
imshow(img_86)
title('given image brain_ 86')
n2=ML_classificatino(img_86,img_86_1,m,v,pw);
text="ML_accuracy percentage of img_86: " + num2str(n2);
disp(text)
subplot(1,2,2)
imshow(img_86_1)
title('given lable image')

figure;
imshow(img_99)
title('given image brain_ 99')
n3=ML_classificatino(img_99,img_99_1,m,v,pw);
text="ML_accuracy percentage of img_99: " + num2str(n3);
disp(text)
subplot(1,2,2)
imshow(img_99_1)
title('given lable image')

figure;
imshow(img_105)
title('given image brain_ 105')
n4=ML_classificatino(img_105,img_105_1,m,v,pw);
text="ML_accuracy percentage of img_105: " + num2str(n4);
disp(text)
subplot(1,2,2)
imshow(img_105_1)
title('given lable image')













%ML estimate
function out=ML_classificatino(img_94,img_94_1,m,v,pw)
m1=m(1);
m2=m(2);
m3=m(3);
v1=v(1);
v2=v(2);
v3=v(3);

k1=double(img_94);
k2=k1;
%k2=img_94_1;
[h,w]=size(k1);

for i=1:h
    for j=1:w
        if k2(i,j)~=0
        px_w1= (1/sqrt(2*pi*v1))*exp((-1/(2*v1))*(k1(i,j)-m1)^2);
        px_w2= (1/sqrt(2*pi*v2))*exp((-1/(2*v2))*(k1(i,j)-m2)^2);
        px_w3= (1/sqrt(2*pi*v3))*exp((-1/(2*v3))*(k1(i,j)-m3)^2);

        p1=[px_w1 px_w2 px_w3];
        p2=max(p1);
        p3 = find(p1==p2);

        if p3==1
            k2(i,j)=85;
        elseif p3==2
            k2(i,j)=170;
        elseif p3==3
            k2(i,j)=255;
        end
        end
    end
end
                
k2;
figure;
subplot(1,2,1)
imshow(k2/255)
title('image after classification')


% error
k11=img_94_1;
count=0;
k2;
for i=1:h
    for j=1:w
        if k11(i,j)==k2(i,j)
            count=count+1;
        end
    end
end
count;
accuracy_percentage= count/numel(k1);
out=accuracy_percentage;

end
















