from moviepy.editor import *


# convert mp4 to gif
clip=(VideoFileClip("Output_video/20220508_21:14:28.mp4"))
clip.write_gif("output.gif")
print("转换完成了")