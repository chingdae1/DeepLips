import glob
from xml.etree.ElementTree import parse
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mpy
import os
import sys

class Parser:
    def __init__(self, all_xml, all_video):
        self.all_xml = all_xml
        self.all_video = all_video
        self.start_list = []
        self.end_list = []
        self.all_text = []

    def parse_dur_and_start(self):
        for file in self.all_xml:
            this_end_list = []
            this_start_list = []
            this_text_list = []
            tree = parse(file)
            note = tree.getroot()
            text_tags = note.findall("text")
            for tag in text_tags:
                dur = float(tag.get('dur'))
                start = float(tag.get('start'))
                start = round(start, 2)
                end = start + dur
                end = round(end, 2)
                this_end_list.append(end)
                this_start_list.append(start)
                this_text_list.append(tag.text)
            self.start_list.append(this_start_list)
            self.end_list.append(this_end_list)
            self.all_text.append(this_text_list)

    def slice_video(self, output_dir):
        print(len(self.start_list))
        for f_idx in range(len(self.all_xml)):
            for i in range(len(self.all_video[f_idx])):
                start = self.start_list[f_idx][i]
                start -= 0.2
                end = self.end_list[f_idx][i]
                origin_file = self.all_video[f_idx]
                myclip = mpy.VideoFileClip(origin_file)
                myclip2 = myclip.subclip(start, end)
                directory = os.path.join(output_dir, str(f_idx))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = str(f_idx) + '_' + str(i)
                new_file_name = os.path.join(output_dir, str(f_idx))
                new_file_name = os.path.join(new_file_name, file_name)

                myclip2.write_videofile(new_file_name + '.mp4', audio=True)
                # ffmpeg_extract_subclip(origin_file, start, end, targetname=new_file_name + '.mp4')

                f = open(new_file_name + '.txt', 'w')
                f.writelines(self.all_text[f_idx][i].replace('\n', ' '))
                f.close


if __name__ == '__main__':
    # path of original video and XML data
    data_path = sys.argv[1]
    # path of result data
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_dir = os.path.join(data_path, '*.xml')
    video_dir = os.path.join(data_path, '*.mp4')

    all_xml = glob.glob(xml_dir)
    all_xml = sorted(all_xml)
    all_video = glob.glob(video_dir)
    all_video = sorted(all_video)

    parser = Parser(all_xml, all_video)
    parser.parse_dur_and_start()
    parser.slice_video(output_dir)
