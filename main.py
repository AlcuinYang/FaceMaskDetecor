import image
import video
from realtime import realtime
import warnings
import cv2

warnings.filterwarnings("ignore")


def mainpage():
    print((' ________Face Mask Detection___________'))
    print('|       please choose model            |')
    print('|        You can choose:               |')
    print('|         1.Image Detection            |')
    print('|         2.video Detection            |')
    print('|         3.Real-time Detection        |')
    print('|         4.exit                       |')
    print('---------------------------------------')
    choose = input("|Select your mode:                     |\n")

    if choose == '1':
        print("You have chosen Image detection")
        address = input("Please input the address of images: \n")
        try:
            print('click q to exit')
            image.tagImg(address)
            # cv2.destroyAllWindows()

        except :
            print("Please check your address")
            mainpage()
    elif choose == '2':
        print("You have chosen Video detection")
        address = input("Please input the address of video: \n")
        try:
            video.tagVideo(address)
        except :
            print("Please check your address")
            mainpage()
    elif choose == '3':
        print("You have chosen real-time detection")
        realtime()

    elif choose == '4':
        exit()
    else:
        print('Please choose correct mode!')
        mainpage()


if __name__ == '__main__':
    mainpage()
