
## XRF Adaptive Signal Processing using Self-supervised Learning

<img width="708" alt="image" src="https://github.com/user-attachments/assets/419abf39-8646-4012-9dd3-944c43afdefa" />
Pipeline used in this paper. 

Macro X-ray fluorescence (XRF) imaging is a popular technique for analyzing historical paintings. XRF imaging reveals the chemical elements present at each pixel, which supports conclusions regarding the pigments used in the painting. Capturing an XRF image is time consuming though, since each pixel’s XRF spectrum is measured by a raster scanning probe
for a constant time per pixel—a time long enough to havean acceptable signal-to-noise ratio. In an effort to accelerate the XRF measurement process, we propose a novel two-stage self-supervised learning framework that allows the dwell time to vary at each pixel. After a quick initial scan, a neural network learns the dwell time per pixel for the second adaptive scan such that (1) the mean squared error of the measurements is minimized and (2) the total scan time does not exceed the time requested by the user. We show under simulations that our method outperforms other sampling techniques.
