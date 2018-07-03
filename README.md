# 第13周作业任务如下：

1. 针对某一场景或物体拍摄相对圆周（圆弧）运动的多角度图片序列（视场景大小和角度范围，一般应有20-60张），学习使用VisualSFM，实现一个3D模型，可在Meshlab下观看。

   ### 实验环境：Windos 10/VisualSFM/MeshLab

   ### VisualSFM

   #### 原始图片（小河滩62张）

   ![](https://ws4.sinaimg.cn/large/006tNc79gy1fsx3o5a9f5j31im0ya4qq.jpg)

   ![](https://ws3.sinaimg.cn/large/006tNc79gy1fsx3p7ml9mj31ii0x67wi.jpg)

   #### 整体步骤

   ![](https://ws1.sinaimg.cn/large/006tNc79gy1fsx3atij9oj30yc0j0acw.jpg)

   #### 导入图片

   ![](https://ws1.sinaimg.cn/large/006tNc79gy1fsx3d8qyqcj30r50f7hdt.jpg) 

   #### 进行sift特征提取和match

   ![](https://ws2.sinaimg.cn/large/006tNc79gy1fsx3eutzxuj30r50f5hdt.jpg)

   #### 稀疏重建/稠密重建

   ![](https://ws3.sinaimg.cn/large/006tNc79gy1fsx3h3c28rj30r50f4hdt.jpg)

   ### MeshLab

   #### 打开文件

   ![](https://ws4.sinaimg.cn/large/006tNc79gy1fsx3iba355j30pc0f4e81.jpg)

   #### import mesh 

   ![](https://ws4.sinaimg.cn/large/006tNc79gy1fsx3j6bbmsj30il0d51i9.jpg)

   #### merge

   ![](https://ws1.sinaimg.cn/large/006tNc79gy1fsx3jlyhpbj30pc0eze81.jpg)

   ![](https://ws2.sinaimg.cn/large/006tNc79gy1fsx3jrvm7lj30pc0f4e81.jpg)

   #### 参数化和纹理投影

   <u>菜单栏->Filter->Texture-> Parameterization + texturing from registered rasters</u>

   ![](https://ws4.sinaimg.cn/large/006tNc79gy1fsx3ljvr3kj311k0j7npe.jpg)

   #### 显示模型

   ![](https://ws1.sinaimg.cn/large/006tNc79gy1fsx3luj09yj311y0k8u0y.jpg)

   

2. 学习opencv+dlib 实现人脸轮廓的检测算法，并用摄像头完成人脸器官的检测，并将脸型、眉毛、鼻子、嘴巴的特征点分别用实线连接；转动人脸或改变面部表情，验证算法的实时性和准确性。

   **项目地址：**https://github.com/zhaoxuyan/Expression-Detection

   **位于** **2. dlib+SVM+gabor中**

   <img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fsry24clq0j30n407ugm8.jpg" width="400px">

3. (选做) 在2的基础上，完成人脸的一种美颜或特效功能，比如描眉、瘦脸、添加猫耳朵等等。此项任务可以作为大作业题目选择之一。

   - 为我的证件照添加**<u>圣诞帽</u>**

   ```python
   #coding=utf-8
   import cv2
   
   # OpenCV人脸识别分类器
   classifier = cv2.CascadeClassifier(
       "/Users/zhaoxuyan/anaconda/pkgs/opencv-3.1.0-np112py27_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
   )
   
   img = cv2.imread("./img/me.png")  # 读取图片
   imgCompose = cv2.imread("./img/hat1.png")
   
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
   color = (0, 255, 0)  # 定义绘制颜色
   # 调用识别人脸
   faceRects = classifier.detectMultiScale(
       gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
   if len(faceRects):  # 大于0则检测到人脸
       for faceRect in faceRects:  
           x, y, w, h = faceRect
           sp = imgCompose.shape
           imgComposeSizeH = int(sp[0]/sp[1]*w)
           if imgComposeSizeH>(y-20):
               imgComposeSizeH=(y-20)
           imgComposeSize = cv2.resize(imgCompose,(w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
           top = (y-imgComposeSizeH-20)
           if top<=0:
               top=0
           rows, cols, channels = imgComposeSize.shape
           roi = img[top:top+rows,x:x+cols]
   
           # Now create a mask of logo and create its inverse mask also
           img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
           ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) 
           mask_inv = cv2.bitwise_not(mask)
   
           # Now black-out the area of logo in ROI
           img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
   
           # Take only region of logo from logo image.
           img2_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask=mask)
   
           # Put logo in ROI and modify the main image
           dst = cv2.add(img1_bg, img2_fg)
           img[top:top+rows, x:x+cols] = dst
   
   # cv2.imshow("image", img) 
   cv2.imwrite("merry-christmas.png", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```

   - 结果

   <img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fsrzhmtzgcj30ic0wswlx.jpg" width="150px">

4. 总结

这次实验分为三个部分：

一是VisualSFM+Meshlab的三维重建，按照网上教程一步一步成功了，效果很好。

第二个是Dlib描人脸，这个我在上一次大作业中已经完成过了。

最后一个是利用OpenCV加装饰品，这个也没啥难度，用Python很简单就解决了。