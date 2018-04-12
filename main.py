from src import *
from PIL import Image


def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# gx, gy, gxlist, gylist = estimate_watermark('images/fotolia_processed')
# gx, gy, gxlist, gylist,num_images = estimate_watermark('images2_processed')
gx, gy, gxlist, gylist,num_images = estimate_watermark('preprocessed')





# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)


plt.figure(3)
plt.imshow(PlotImage(cropped_gx))


alph_image = MatrixToImage(PlotImage(cropped_gx))
alph_image.save('cropped_gx.jpg')

plt.figure(4)
plt.imshow(PlotImage(cropped_gy))
plt.show()


W_m = poisson_reconstruct(cropped_gx, cropped_gy)
#
plt.figure()
plt.imshow(PlotImage(W_m))
plt.show()


# random photo
# img = cv2.imread('images/fotolia_processed/fotolia_137840645.jpg')
# img = cv2.imread('images/fotolia_processed/fotolia_69189206.jpg')
img = cv2.imread('preprocessed/0b76b4df-4bb2-4e3c-9173-c05153367d61.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

plt.imshow(im)
plt.show()



# We are done with watermark estimation
# W_m is the cropped watermark
# num_images = len(list(gxlist))

# J, img_paths = get_cropped_images('images2_processed', num_images, start, end, cropped_gx.shape)
J, img_paths = get_cropped_images('preprocessed', num_images, start, end, cropped_gx.shape)
# get a random subset of J
idx = [389, 144, 147, 468, 423, 92, 3, 354, 196, 53, 470, 445, 314, 349, 105, 366, 56, 168, 351, 15, 465, 368, 90, 96, 202, 54, 295, 137, 17, 79, 214, 413, 454, 305, 187, 4, 458, 330, 290, 73, 220, 118, 125, 180, 247, 243, 257, 194, 117, 320, 104, 252, 87, 95, 228, 324, 271, 398, 334, 148, 425, 190, 78, 151, 34, 310, 122, 376, 102, 260]
idx = idx[:25]
# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# plt.figure()
# plt.imshow(PlotImage(Wm))
# plt.show()


# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm,num_images=4)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)

alph_image = MatrixToImage(PlotImage(alph))
alph_image.save('alph.jpg')
alph_image = MatrixToImage(PlotImage(Wm))
alph_image.save('Wm.jpg')


C, est_Ik = estimate_blend_factor(J, Wm, alph)
# plt.figure('alph')
# plt.imshow(PlotImage(alph))
# plt.show()


alph_image = MatrixToImage(PlotImage(est_Ik))
alph_image.save('est_Ik.jpg')


# plt.figure('est_Ik')
# plt.imshow(PlotImage(est_Ik))
# plt.show()




alpha = alph.copy()
for i in range(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]
# plt.figure('alpha')
# plt.imshow(PlotImage(alpha))
# plt.show()

alph_image = MatrixToImage(PlotImage(alpha))
alph_image.save('alpha1.jpg')



Wm = Wm + alpha*est_Ik
alph_image = MatrixToImage(PlotImage(Wm))
alph_image.save('Wm1.jpg')
W = Wm.copy()
for i in range(3):
	W[:,:,i]/=C[i]

# Jt = J[:25]
Jt = J[:1]
# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
alph_image = MatrixToImage(PlotImage(Wk))
alph_image.save('Wk.jpg')
alph_image = MatrixToImage(PlotImage(Ik))
alph_image.save('Ik.jpg')
alph_image = MatrixToImage(PlotImage(W))
alph_image.save('W.jpg')
alph_image = MatrixToImage(PlotImage(alpha1))
alph_image.save('alpha1.jpg')
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  

