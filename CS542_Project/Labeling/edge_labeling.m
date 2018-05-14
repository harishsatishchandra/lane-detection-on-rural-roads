
% Please change the variables named ‘path’, ‘imagefiles’,  ‘dest’, 
% as well as the input parameter for the mkdir function at the beginning 
% of the code. The path variable is used to create the full name of the 
%image. Imagefiles variable is a struct, containing all the images in 
% the directory. Dest variable contains the path to the destination 
%folder, where all the output images are stored. The images needed to 
%run this code can be found in the google drive link, under the name:  
clc; clear all; close all;
path = '/home/harish-admin/Downloads/2011_09_26/2011_09_26_drive_0019_sync/image_02/data/';
imagefiles = dir('/home/harish-admin/Downloads/2011_09_26/2011_09_26_drive_0019_sync/image_02/data/*.png');
mkdir('/home/harish-admin/Downloads/edge_images1');
dest = '/home/harish-admin/Downloads/edge_images1/';
len = length(imagefiles);
for ii = 1:len
    currentfilename = strcat(path,imagefiles(ii).name);
    f=imread(currentfilename);
    ycb = rgb2ycbcr(f);
    ycb_1 = ycb(:,:,1);
    hist = histeq(ycb_1);
    g=double(rgb2gray(f));
    c=0.3;
    fc = sqrt(double(f(:,:,1).^2)+double(f(:,:,2).^2)+double(f(:,:,3).^2));
    fr = double(f(:,:,1))./fc;
    fg = double(f(:,:,2))./fc;
    fb = double(f(:,:,3))./fc;
    inv1 = cat(3,fr,fg,fb);
    [M,N]=size(g);
    for x = 1:M
        for y = 1:N
            m=double(g(x,y));
            z(x,y)= log(1+m);
        end
    end
    for x = 1:M
        for y = 1:N
            m=double(inv1(x,y));
            z1(x,y)= log(1+m);
        end
    end
    img = z;
    inv1_g = z1;
    [H,W,C] = size(img);
    img = double(img); 
    inv2 = 0.4 - (cos(pi/3).*(log(double(inv1(:,:,1))./...
        double(inv1(:,:,2)))))+(sin(pi/3).*(log(double(inv1(:,:,3))./...
        double(inv1(:,:,2)))));
    inv2 = abs(inv2).*255;
    inv2 = uint8(inv2);
    img1 = uint8(img);
    img1 = imgaussfilt(img1);
    inv3 = medfilt2(inv2, [15 15]);
    inv4 = medfilt2(hist, [15 15]);
    for i = 1:M
        for j = 1:N
            if inv3(i,j) > 180
                inv3(i,j) =95;
            else
                inv3(1,j) = 1;

            end
        end
    end
    for i = 1:M
        for j = 1:N
            if inv4(i,j) > 180
                inv4(i,j) =95;
            else
                inv4(1,j) = 1;

            end
        end
    end
    se = strel('line',11,0);
    se1 = strel('line',19,0);
    dilate = imdilate(inv3,se);
    dilate1 = imdilate(inv4,se);    
    for i = 1:M
        for j = 1:N
            if dilate(i,j)  >= 80
                dilate(i,j) = 75;            
            end
        end
    end
    for i = 1:M
        for j = 1:N
            if dilate1(i,j)  >= 80
                dilate1(i,j) = 75;
            
            end
        end
    end
    erode = imerode(dilate,se);
    erode1 = imerode(dilate1,se);    
    inv5 = edge(erode,'canny',[0.15 0.2]);
    inv5 = imcrop(inv5, [1 200 1242 400]);
    inv6 = edge(erode1,'canny',[0.13 0.2]);
    inv6 = imcrop(inv6, [1 200 1242 400]);
    inv7 = inv5 + inv6;
    fulldestination = fullfile(strcat(dest,imagefiles(ii).name));
    imwrite(inv7, fulldestination);
end