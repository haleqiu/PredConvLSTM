__author__ = 'Hale Qiu'
import tensorflow as tf
import os
import numpy as np


class Kitti_Dataset(object):
    def __init__(self, data_dir = None, batch_size = 8, time_step = 7, chanels = 3,unique = "False",video_gap=1):
        
        self.data_dir = data_dir
        self.file_list = []
        self.label_list = []
        self.dataset = None
        self.batch_size = batch_size
        self.train_images = None
        self.test_images = None
        self.validation_images = None
        self.time_step = time_step
        self.next_batch = None
        self.upper_bound = 10000
        self.epho_size = 0
        self.unique = unique
        self.video_gap = video_gap
    
    def overset(self):
        if self.data_dir:
            self._read_files()
            print("initialize the KITTI dataset for over")
                                                    
            filenames = tf.constant(self.file_list[1:40])
            labels = tf.constant(self.label_list[1:40])
            kitti_data = tf.data.Dataset.from_tensor_slices((filenames, labels))

            kitti_data = kitti_data.map(self._par_list_fn)
            kitti_data = kitti_data.shuffle(1000).repeat().batch(self.batch_size)
            self.dataset = kitti_data
            iterator = kitti_data.make_one_shot_iterator()
            self.next_batch = iterator.get_next()
            self.epho_size = int(len(self.label_list[1:40])/self.batch_size)
        return kitti_data
    
    def predset(self):
        if self.data_dir:
            self._read_files()
            print("initialize the KITTI dataset for pred")
                                                    
            filenames = tf.constant(self.file_list)
            labels = tf.constant(self.label_list)
            kitti_data = tf.data.Dataset.from_tensor_slices((filenames, labels))

            kitti_data = kitti_data.map(self._par_list_fn)
            kitti_data = kitti_data.shuffle(buffer_size=1000).repeat(1000).batch(self.batch_size)
            self.dataset = kitti_data
            iterator = kitti_data.make_one_shot_iterator()
            self.next_batch = iterator.get_next()
            self.epho_size = int(len(self.label_list)/self.batch_size)
        return kitti_data
    
    
    def vaeset(self):
        if self.data_dir:
            self._read_files()
            print("initialize the KITTI dataset for vae")
            #rawfile = set(self.file_list).union(set(self.label_list))
            self.rawfile = list(np.reshape(self.file_list,(-1))) +self.label_list
            
            kitti_data = tf.data.Dataset.from_tensor_slices(self.rawfile)
            kitti_data = kitti_data.map(self._parse_function)
            kitti_data = kitti_data.shuffle(buffer_size=1000).repeat(1000).batch(self.batch_size)
            self.dataset = kitti_data
            iterator = kitti_data.make_one_shot_iterator()
            self.next_batch = iterator.get_next()
            self.epho_size = int(len(self.label_list)*self.time_step/self.batch_size)
            
        return kitti_data
        
    #def _read_files(self, data_dir = self.data_dir, time_step = self.time_step):
    def _read_files(self):
        # append all the training data files directory into filelist and labelist
        image_folder = os.listdir(self.data_dir)
        filename_path = []
        count = 0
        for folder in image_folder:
            count = count +1
            if count==self.upper_bound:
                break
            if os.path.isdir(self.data_dir+'/'+folder):
                filename_path.append(folder)

        file_name = [self.data_dir+ '/' + x + '/' for x in filename_path]
        file_list, label_list = [], []
        
        if not self.unique:
            for f in file_name:
                t = self.time_step * self.video_gap 
                while os.path.exists(f + "%06d"%t + ".png"):
                    file_list.append([f + "%06d"%i + ".png" for i in range(t-(self.time_step)*self.video_gap,t-self.video_gap, self.video_gap)])
                    label_list.append(f + "%06d"%t + ".png")
                    t += self.time_step * self.video_gap 
        else:
            for f in file_name:
                t = self.time_step * self.video_gap 
                while os.path.exists(f + "%06d"%t + ".png"):
                    file_list.append([f + "%06d"%i + ".png" for i in range(t-(self.time_step)*self.video_gap,t-self.video_gap, self.video_gap)])
                    label_list.append(f + "%06d"%t + ".png")
                    t += 1
                    
        self.file_list = file_list
        self.label_list = label_list
        
    @property
    def sample(self):
        sample = cv2.imread(self.file_list[0])
        return sample

    @property
    def labels(self):
        sample = cv2.imread(self.label_list[0])
        return sample

    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    def _parse_function(self, filename):

        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        image_crop = tf.image.crop_to_bounding_box(image_decoded,0,484,370,370)
        image_resized = tf.image.resize_images(image_crop, [256, 256])
        #return image_decoded
        return image_resized

    
    def _par_list_fn(self, file_list, label_name):
        file_shape = file_list.get_shape().as_list()
        image_resized = tf.stack([self._parse_function(file_list[i]) for i in range(file_shape[0])])
        label = self._parse_function(label_name)
        return image_resized, label

