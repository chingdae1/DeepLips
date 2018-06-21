import glob
import sys
import os
import cv2
import shutil

class Categorizer:
    def __init__(self, all_video, all_text, output_path, boundary):
        self.all_video = all_video
        self.all_text = all_text
        self.output_path = output_path
        self.boundary = boundary

    def categorize(self):
        for idx in range(len(all_video)):
            cap = cv2.VideoCapture(all_video[idx])
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dir_name = str(int(frames/100) * 100)
            new_dir_path = os.path.join(self.output_path, dir_name)
            if os.path.exists(new_dir_path) is False:
                os.mkdir(new_dir_path)

            new_file_name = all_video[idx].split('/')[-2] + '_' + all_video[idx].split('/')[-1].split('.')[0]
            new_video_path = os.path.join(new_dir_path, new_file_name + '.mp4')
            new_text_path = os.path.join(new_dir_path, new_file_name + '.txt')
            print('Copy ' + new_file_name + '.mp4 [' + str(idx) + '/' + str(len(all_video)) + ']')
            shutil.copy2(all_video[idx], new_video_path)
            shutil.copy2(all_text[idx], new_text_path)

    def print_distribution(self, dir_path):
        print('')
        print('---------File Distribution---------')
        print('[directory_name] : [number of mp4 files]')

        all_dir = glob.glob(os.path.join(dir_path, '*'))
        for dir in all_dir:
            list = os.listdir(dir)
            number_files = len(list)
            dir_name = dir.split('/')[-1]
            print(dir_name, ':', int(number_files/2), 'mp4 files')


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    boundary = sys.argv[3]

    video_path = os.path.join(data_path, '*', '*', '*.mp4')
    text_path = os.path.join(data_path, '*', '*', '*.txt')
    all_video = sorted(glob.glob(video_path))
    all_text = sorted(glob.glob(text_path))

    categorizer = Categorizer(all_video, all_text, output_path, boundary)
    categorizer.categorize()
    categorizer.print_distribution(output_path)


