"""
获取图片数据和对图片进行分组处理
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
 
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
  
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
 
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
  
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
     
    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
 
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
  
    ######################################
    # data argumentation should go to here
    ######################################
 
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
 
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉75行（标准化）和 95行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)
  
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size = batch_size,
                                                num_threads = 64, 
                                                capacity = capacity)

    # you can also use shuffle_batch 
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == "__main__":
    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 208
    IMG_H = 208

    train_dir = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/data/train/'

    image_list, label_list = get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:

                img, label = sess.run([image_batch, label_batch])

                # test one batch
                for j in np.arange(BATCH_SIZE):
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
