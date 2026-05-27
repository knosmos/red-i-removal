import numpy as np
import cv2

FILE = "train_data_batch_1"
data = np.load(FILE, allow_pickle=True)
print(data.keys())
print(data['data'][0])

# reshape (64 x 64 x 3)
test_img = data['data'][1].reshape(3, 64, 64)
test_img = np.transpose(test_img, (1, 2, 0))
test_img = cv2.resize(test_img, (256, 256))
cv2.imshow("test", test_img)
cv2.waitKey(0)