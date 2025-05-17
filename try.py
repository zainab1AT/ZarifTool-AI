#with hips
#normal 0.10828632116317749
#normal 0.03372156620025635
#normal 0.06198093295097351
#normal 0.06449726223945618

#with shoulders
#normal 0.13527962565422058
#normal 0.07971012592315674
#normal 0.09741970896720886
#normal 0.1331009566783905

#with shoulders
#unnormal 0.0834883451461792
#unnormal 0.10949325561523438 r
#unnormal 0.09065866470336914 r
#unnormal 0.16437172889709473 r

#with hips
#unnormal 0.15863576531410217
#unnormal 0.06221318244934082 r
#unnormal 0.059503793716430664
#unnormal 0.21135073900222778


import numpy as np

# Normal and forward head posture values
normal = [0.018077492713928223, 0.11108490824699402]
unnormal = [0.04863739013671875]

# Calculate mean and standard deviation
normal_mean = np.mean(normal)
normal_std = np.std(normal)
unnormal_mean = np.mean(unnormal)
unnormal_std = np.std(unnormal)

print("Normal Posture Mean:", normal_mean)
print("Normal Posture Std Dev:", normal_std)
print("Unnormal Posture Mean:", unnormal_mean)
print("Unnormal Posture Std Dev:", unnormal_std)

threshold = (normal_mean + unnormal_mean) / 2
print("Calculated Threshold:", threshold)


# print(0.4448063373565674 / 0.2784733176231384) 


"""
ear neck distance 1.1776956
ear shoulder distance 1.0871085
forward head? False
Saved image with keypoints to physioOutImages/test285_1.18.jpg
imagesTakenC(285).jpg -------- NOT

ear neck distance 1.1241809
ear shoulder distance 1.0911028
forward head? False
Saved image with keypoints to physioOutImages/test282_1.12.jpg
imagesTakenC(282).jpg -------- NOT

ear neck distance 1.1126363
ear shoulder distance 1.0584438
forward head? False
Saved image with keypoints to physioOutImages/test302_1.11.jpg
imagesTakenC(302).jpg -------- NOT

ear neck distance 1.1568944
ear shoulder distance 1.1426228
forward head? False
Saved image with keypoints to physioOutImages/test312_1.16.jpg
imagesTakenC(312).jpg -------- NOT


ear neck distance 1.0984603
ear shoulder distance 1.0261338
forward head? False
Saved image with keypoints to physioOutImages/test5_1.10.jpg
imagesTakenC(5).jpg -------- normal

ear neck distance 1.0764356
ear shoulder distance 1.0026318
forward head? False
Saved image with keypoints to physioOutImages/test18_1.08.jpg
imagesTakenC(18).jpg -------- normal

ear neck distance 1.0954125
ear shoulder distance 1.0210809
forward head? False
Saved image with keypoints to physioOutImages/test23_1.10.jpg
imagesTakenC(23).jpg -------- normal

ear neck distance 1.0854435
ear shoulder distance 1.0538222
forward head? False
Saved image with keypoints to physioOutImages/test185_1.09.jpg
imagesTakenC(185).jpg -------- normal

"""


"""
ear neck distance 1.1332796
ear shoulder distance 0.97521365
forward head? False
Saved image with keypoints to physioOutImages/test0_1.13.jpg
testtt (1).jpeg ---------- normal

ear neck distance 1.1934057
ear shoulder distance 1.1268697
forward head? False
Saved image with keypoints to physioOutImages/test1_1.19.jpg
testtt (2).jpeg ---------- normal

ear neck distance 1.3919656
ear shoulder distance 1.0585064
forward head? True
Saved image with keypoints to physioOutImages/test2_1.39.jpg
testtt (3).jpeg ---------- normal

ear neck distance 1.2838311
ear shoulder distance 1.2755201
forward head? True
Saved image with keypoints to physioOutImages/test3_1.28.jpg
testtt (4).jpeg ---------- NOT!!

ear neck distance 1.26644
ear shoulder distance 1.2110498
forward head? True
Saved image with keypoints to physioOutImages/test4_1.27.jpg
testt (1).jpeg ---------- NOT!!

ear neck distance 1.3070753
ear shoulder distance 1.2884226
forward head? True
Saved image with keypoints to physioOutImages/test5_1.31.jpg
testt (2).jpeg ---------- NOT!!

ear neck distance 1.1985717
ear shoulder distance 1.0209312
forward head? False
Saved image with keypoints to physioOutImages/test6_1.20.jpg
testt (3).jpeg ---------- normal!!


"""
