

import os
import numpy as np
import tensorflow as tf
import input_data
import model


N_CLASSES = 2
IMG_W = 208  # resize the image
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000 #train step
learning_rate = 0.0001 #


def run_training():
   
    train_dir = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/data/train/'
    logs_train_dir = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/logs/'
   
    train, train_label = input_data.get_files(train_dir)
   
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY) 
   
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
     
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
   
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
              
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
           
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    run_training()
