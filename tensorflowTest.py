
from src import *
from src import tensorflow_experiments
import tensorflow as tf
from PIL import Image




def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



gx, gy, gxlist, gylist,num_images = estimate_watermark('images/fotolia_processed')

# est, loss = poisson_reconstruct(gx, gy)
est = poisson_reconstruct(gx, gy)
cropped_gx, cropped_gy = crop_watermark(gx, gy)
est2 = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread('images/fotolia_processed/fotolia_26734091.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)



plt.figure(figsize=(12, 12), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(im)


print(cropped_gx.shape, cropped_gy.shape, est.shape, est2.shape)
print(im.shape, start, end)

plt.imshow(img[start[0]:(start[0]+end[0]), start[1]:(start[1]+end[1]), :])



'''
This is the part where we get all the images, extract their parts, and then add it to our matrix
'''

images_cropped = np.zeros((num_images,) + cropped_gx.shape)

# get images
foldername = 'images/fotolia_processed'

# Store all the watermarked images
# start, and end are already stored
# just crop and store image
image_paths = []
_s, _e = start, end
index = 0

# Iterate over all images
for r, dirs, files in os.walk(foldername):

    for file in files:
        _img = cv2.imread(os.sep.join([r, file]))
        if _img is not None:
            # estimate the watermark part
            image_paths.append(os.sep.join([r, file]))
            _img = _img[_s[0]:(_s[0]+_e[0]), _s[1]:(_s[1]+_e[1]), :]
            # add to list images
            images_cropped[index, :, :, :] = _img
            index+=1
        else:
            print("%s not found."%(file))


images_cropped=images_cropped[5:]
num_images, m, n, chan = images_cropped.shape
model = tensorflow_experiments.image_watermark_decompose_model(num_images, m, n, chan)

# define the variables
# plt.imshow(PlotImage(est2))
W_m = est2
J = images_cropped
I = np.random.randn(num_images, m, n, chan)
alpha = np.random.rand(m, n, chan)
W_median = W_m.copy()

alph_image = MatrixToImage(PlotImage(W_m))
alph_image.save('W_mt.jpg')







# W_m_copy = np.random.randn(est2.shape[0],est2.shape[1],est2.shape[2])
# W =np.stack([np.random.randn(est2.shape[0],est2.shape[1],est2.shape[2]) for _ in range(num_images)])
# W = np.stack([W_m for _ in range(num_images)])
print
saver = tf.train.Saver()

tf.initialize_all_variables()
# tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print("Start")
        _, loss = sess.run([model['step'], model['loss']], feed_dict={
            model['J']: J,
            model['alpha']: alpha,
            model['W_m']: W_m,
            model['W_median']: W_median,
            # model['I']: I,
            # model['W']: W
        })
        print(loss)