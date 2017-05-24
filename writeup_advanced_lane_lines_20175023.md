## Advanced Lane Finding Project
### Project 4

---

In the first project for this nanodegree a simple approach to finding lane lines was used. Canny edge detection, Hough transforms, and linear regression were used to accomplish the task of finding approximately where the lane lines in the provided sample video were.

For this project, an approach that relies on more advanced methods is used. These approaches are:
- calibrating the camera
- correcting distortion
- using color transforms and gradients to create a threshold binary image
- applying perspective transforms
- increasing the order of the model using to detect the curvature of the lane lines

The pipeline that will be built on these methods is as follows:
1. Compute the camera calibration matrix and determine the distortion coefficients based on chessboard images.
2. Apply the distortion correction to the raw images.
3. Use the color and gradient thresholding to create a binary image.
4. Apply a perspective transform to aid in determine the curvature of the lane lines.
5. Detect the lane line pixels and then the entire lane line.
6. Determine the curvature of each (left and right) lane line and the vehicle position with respect to the center of the lane.
7. Display the identified lane lines on the original images and combine to create a video with the estimated curvature and vehicle offset from center.

[//]: # (Image References)

[image1]: ./output_images/output_9_1.png "Calibration"
[image2]: ./output_images/output_11_0.png "Undistorted"
[image3]: ./output_images/straight_lines2.jpg "Original"
[image4]: ./output_images/output_11_1.png "Undistorted"
[image5]: ./output_images/output_11_1.png "Undistorted"
[image6]: ./output_images/output_21_1.png "Combined Binary"
[image7]: ./output_images/output_26_1.png "Original and Warped Straight"
[image8]: ./output_images/output_28_1.png "Original and Warped Curved"
[image9]: ./output_images/output_45_1.png "Overlay"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the Jupyter notebook under the heading 'Calibrate camera and correct distortion'. The method for camera calibration shown in the classroom provided a template for calibration in this project. As such, the methods from the classroom were wrapped in functions so that they could be used in a pipeline.

calibrate() starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. It is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Images for which corners were found were appended to calibration_success[] in order to view later. The calibration images for which corners were found are shown below:

![Calibration Images][image1]

Then, the function calibrate_and_undistort(), used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Undistorted][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The test images were distortion-corrected using the method described above. Below, the original image and distortion-corrected image are shown together.

![Original][image3]
![Undistorted][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

We were shown several different methods to detect the lane lines in an image. At the heart of these methods is the Sobel operator. Applying the Sobel operator allows us to take the derivative of the image in the x or y direction. Using different combinations of the Sobel operator we can mask the input images based on the gradient absolute value, gradient direction, and gradient magnitude. By combining these different masks we can isolate the lane lines in each image. Applying a color mask afterwards further emphasizes the lane lines.

The first two code blocks below the 'Color and Gradient Thresholding' show how each of the 'masks' were created. I used a combination of the all of the above gradient thresholds as well as multiple color thresholds. I am not completely satisfied with the results but it seems to work well enough for project_video.mp4.

![Undistorted][image5]
![Combined Binary][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the first code block after the 'Perpective Transform' heading. The function takes an input image and determines the src and dst coordinates based on the size of the image and a chosen offset. The code for the src and dst coordinates is below:

```python
    offset = 100.
    # get height and width of image
    h, w = image.shape[0], image.shape[1]

    # specify source coordinates
    # the height coefficient is very sensitive. <.635 and the transform is not
    # rotated far enough (i.e. looks almost undistorted as you approach 0).
    # >.635 and the image is rotated to far.
    src = np.float32([[w // 2 - offset, h * .635], [w // 2 + offset, h * .635], [-offset, h], [w + offset, h]])

    # specify destination coordinates
    dst = np.float32([[offset, 0], [w - offset, 0], [offset, h], [w - offset, h]])
```

I verified that my perspective transform was working as expected by plotting the original image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Original and Warped Straight][image7]

In addition, I check the curved lane line test images as well to verify that they remained equidistant.

![Original and Warped Curved][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I identified the lane line pixels using the method provided to us in the classroom. This is shown in the code blocks beneath the heading 'Lane Line Tracking'. find_lines() takes a binary, warped image and blindly searches for the lane line pixels. find_lines_post() is called when the position of the lane lines are known. A search within the specified margin is used rather than blindly searching.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The functions calc_radius() and calc_offset() calculate the radius of curvature and the offset from the center of the lane, respectively. The code for calculating the radius of curvature is shown below:

```python
  y_eval = binary_warped.shape[0] - 1
  # Define conversions in x and y from pixels space to meters
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension

  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  # Calculate the new radius of curvature in meters
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this using the functions overlay_lane() and overlay_text(). A test image with the lane and text overlaid is shown below:

![Overlay][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The full pipeline is implemented in the AdvancedPipeline() class.

Here's a [link to my video. ](./P4_Output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When testing my pipeline on the challenge and harder challenge video, it's obvious that my process could be improved. It fails after the first few seconds in the challenge video and fails to start in the harder challenge video. The gradient and color thresholding could definitely be improved to isolate the lane lines more prominently. The thresholds that I chose seem to work sufficiently in the sunlight but fail in the shadows. In addition, the method for finding and tracking lane line pixels could be optimized, however, I did not have time for this project to modify the method provided to us in the classroom. Overall, I think that the pipeline I created is a good baseline but has room for a lot of improvement.
