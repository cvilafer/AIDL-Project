# Test Results for MS-ASL 1000

In the file "Test_Results_MS_ASL_1000.txt" you can see the final Test results for MS-ASL 1000

In the folder "test_frames_img_1000_mlp2_avg_class" there are all the images of the best frames selected by the model.

In the image file name (for example: 00018-cashier-athlete-fr8-p0.00MLP1.png) you can see: the image id, the correct sign, the infered sign, the frame number, the probability/confidence (0-1) from the inference and wether is the best frame from MLP1 or MLP2.
Really, at the end the model chooses the sign from MLP1 or MLP2 with the higher probability.

In the arXiv article, when explaining the Key Frame MLP model, the figure (mlp1_mlp2_examples.png) may not have been the most appropriate choice. But in the folder "test_frames_img_1000_mlp2_avg_class" there are all the test best frames selected by MLP1 and MLP2.

![MLP1/MLP2 Best frames](mlp1_mlp2_examples2.png)
