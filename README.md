# MonReader Project

- MonReader is a new mobile document digitalization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

# Data Description:

- Page flipping video from smart phones labelled as flipping and not flipping.
- Videos were clipped as short videos and labelled as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

# Goal:

- Using a custom CNN model, predict if the page is being flipped using a single image.
- Using a pre-trained ResNet, VGG16, and MobileNet, predict if the page is being flipped using a single image.

# Success Metric:

- Evaluate model performance based on F1 score, the higher the better but should be higher than 91%.
- Model should also have a final size lower than 40Mb so it can fit in a smartphone app.

# Bonus:

- Predict if a given sequence of images contains an action of flipping.

#### Project code: QN49hk5Qub80C76X
