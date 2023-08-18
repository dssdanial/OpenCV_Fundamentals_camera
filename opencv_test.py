import cv2 as cv
import sys
import matplotlib.pyplot as plt

def read_img():
    img = cv.imread("opencv/samples/data/starry_night.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    return img
def show_img():
    img=read_img()
    cv.imshow("Display window", img)
    k = cv.waitKey(0)
    if k==27: #space key
        cv.destroyAllWindows()
    elif k == ord("s"): #read the "s" key
        cv.imwrite("starry_night.png", img)
        cv.destroyAllWindows()

def take_photo():
    cap=cv.VideoCapture(0)

    while(cap.isOpened()):

        ret, frame = cap.read()

        plt.figure(1), cv.imshow("original", frame)

        if cv.waitKey(2) & 0xFF==ord('p'):

            frame1=frame
            print("Captured")
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # plt.figure(0), cv.imshow("gray-scale", gray)

        elif cv.waitKey(2) & 0xFF ==ord('q'):
            print("saving...")
            cv.imwrite("pic.png",frame1)
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def save_video_from_camera():
    cam_port=0
    cap=cv.VideoCapture(cam_port)
    fourcc=cv.VideoWriter_fourcc('M','P','E','G')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    FPS=float(30.0)
    out=cv.VideoWriter("out.avi", fourcc, FPS, (width, height) )


    while(cap.isOpened()):

        ret, frame=cap.read()

        if ret is True:
            cv.imshow("video", frame)
        elif():
            break

        if cv.waitKey(1) & 0xFF ==ord("q"):
            break

        if cv.waitKey(1) & 0xFF ==ord("s"):
            print("Recording...")
            out.write(frame)
    cap.release()
    cv.destroyAllWindows()



if __name__=='__main__':
    #reading and writing image
    # read_img()
    # show_img()
    take_photo()
    #save_video_from_camera()
