clear;clc;
run('./train-s1/cnn_dehaze.m');
run('./train-s2/cnn_dehaze_greedy.m');
run('./train-s2/cnn_dehaze_joint.m');
run('./train-s3/cnn_dehaze_greedy.m');
run('./train-s3/cnn_dehaze_joint.m');