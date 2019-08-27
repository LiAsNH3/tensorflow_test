import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


import input_data
import model


def evaluate_all_image():
    test_dir = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/data/test/'

    N_CLASSES = 2
    IMG_W = 208  # resize the image
    IMG_H = 208
    BATCH_SIZE = 16
    CAPACITY = 2000
    print("-"*50)
    test, test_label = input_data.get_files(test_dir)
    BATCH_SIZE = len(test)
    print("There are %d test images totally....." % BATCH_SIZE)

    print("-"*50)
    test_batch, test_label_batch = input_data.get_batch(test,
                                                        test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
    testloss = model.losses(logits, test_label_batch)
    testacc = model.evaluation(logits, test_label_batch)

    logs_train_dir = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/logs/'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Reading chckpoint.....")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loading success, global steps is %s" % global_step)
        else:
            print("NO such checkpoint file")
        print("-"*50)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_loss, test_acc = sess.run([testloss, testacc])
        print("The model \'s loss is %.2f" % test_loss)
        correct = int(BATCH_SIZE*test_acc)
        print("Correct: %d" % correct)
        print("Wrong: %d" % (BATCH_SIZE - correct))
        print("The accuracy in test images are %.2f%%" % (test_acc*100))
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    evaluate_all_image()
