import numpy as np
import cv2
import time

index_bool = False
serial_bool = False

Color_lower = np.array([110, 50, 50])
Color_upper = np.array([130, 255, 255])
blue_lower = Color_lower
blue_upper = Color_upper

def Camera_init():
    cap.set(3, 320)
    cap.set(4, 240)
try:
    cap = cv2.VideoCapture(1)
    index_bool = True
    Camera_init()
except:
    index_bool = False
    print("摄像头异常.....")


def Start_Program_serial(bool_set):
    if bool_set == True:
        if index_bool == True:
            while True:
                ret, frame = cap.read()
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                #frame = frame[80:320, 85:700]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, blue_lower, blue_upper)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                res = cv2.bitwise_and(frame, frame, mask=mask)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                if len(cnts) > 0 and len(cnts) < 40:
                    cnt = max(cnts, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(frame, (int(x), int(y)), int(radius),(255, 255, 67), 2)
                    x = str(int(x))
                    y = str(int(y))
                    string = x+","+y
                    print(string)
                    print('---')#x-185,y-134
                    print("X_position:",str(x),"Y_position:",str(y))
                    time.sleep(0.06)
                else:
                    pass
                cv2.imshow('frame', frame)
                cv2.imshow('mask', mask)
                cv2.imshow('res', res)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            #-----------END------------
            cap.release()
            cv2.destroyAllWindows()
        elif index_bool == False:
            print("--请检查摄像头连接----------------")
        else:
            print("程序异常,请尝试重启...")
    elif bool_set == False:
        pass
    else:
        print("<attention:>请注意修改主函数的bool_set值...")

def Start_Program_no_serial(bool_set):
    if bool_set == True:
        if index_bool == True:
            while True:
                ret, frame = cap.read()
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                #frame = frame[80:320, 85:700]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, blue_lower, blue_upper)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                res = cv2.bitwise_and(frame, frame, mask=mask)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts) > 2 and len(cnts) < 40:
                    cnt = max(cnts, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(frame, (int(x), int(y)), int(radius),(255, 255, 67), 2)
                    x = str(int(x))
                    print('-')
                    print("X_position:",str(x),"Y_position:",str(y))
                    time.sleep(0.05)
                else:
                    pass
                cv2.imshow('frame', frame)
                cv2.imshow('mask', mask)
                cv2.imshow('res', res)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            #-----------END------------
            cap.release()
            cv2.destroyAllWindows()
        elif index_bool == False:
            print("--请检查摄像头连接----------------")
        else:
            print("程序异常,请尝试重启...")
    elif bool_set == False:
        pass
    else:
        print("<attention:>请注意修改主函数的bool_set值...")

if __name__ == "__main__":

    if serial_bool == True:
        Start_Program_serial(True)#bool_set
    elif serial_bool == False:
        Start_Program_serial(True)
    else:
        print("ERROR")