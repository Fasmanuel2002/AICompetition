# AICompetition
KaggleCompetition HSLU

BFRBS, its about self-directed habits involving repetitive actions, when frequent or intense, can cause physical harm and psychosocial challanges. For this the Child Mind has developed a wrist-worn device "Helios", desinged to detect these behaviors. It has 5 sensors, including 5 thermopiles(for detecting body heat) and 5 time-of-flight sensors(for detecting proximity).

The kids started to do geastures wearing the Helios device 

1.They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

2.They followed this with a short pause wherein they did nothing (Pause); and

3.Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).


The kids on the experiment has 18 unique gestures(8 BFRB == damaging and 10 non-BFRB == no damaging) 



This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device 


The Evaluation will have F1 score as the most important thing 
-Binary F1 on whether the gesture is one of the target or non-target types.
-Macro F1 on gesture, where all non-target sequences are collapsed into a single non_target class

In this competition you will use sensor data to classify body-focused repetitive behaviors (BFRBs) and other gestures.

This dataset contains sensor recordings taken while participants performed 8 BFRB-like gestures and 10 non-BFRB-like gestures while wearing the Helios device on the wrist of their dominant arm. The Helios device contains three sensor types:

-1x Inertial Measurement Unit (IMU; BNO080/BNO085): An integrated sensor that combines accelerometer, gyroscope, and magnetometer measurements with onboard processing to provide orientation and motion data.
-5x Thermopile Sensor (MLX90632): A non-contact temperature sensor that measures infrared radiation.
-5x Time-of-Flight Sensor (VL53L7CX): A sensor that measures distance by detecting how long it takes for emitted infrared light to bounce back from objects.




==============================================================

*DataSet*
