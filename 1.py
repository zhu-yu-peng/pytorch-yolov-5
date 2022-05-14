import cv2
import mediapipe as mp
import time

# 从摄像头捕获视频
# 参数为0表示打开计算机内置摄像头（我的电脑有点带不动，很卡） 也可以用视频的路径
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 如果使用函数本身的预设值的话，括号内不用写参数设置 按住ctrl点击函数可以看到函数的定义
# 画手上landmarks的点的坐标
mpDraw = mp.solutions.drawing_utils
# 点的样式 三个值分别是b、g、r
handLms_Style = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
# 线的样式
handCon_Style = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=10)
# 设置两个时间 为了计算fps：每秒传输帧数
pTime = 0
cTime = 0

while True:
    # 会返回两个值ret和img
    ret, img = cap.read()
    if ret:  # 这表示如果ret没有问题，也就是ret = true
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式的图片转换成RGB的格式
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)      #映射出手上的21个点的坐标
        # 为了下面得到的坐标从比例转换为真实坐标，这里首先得到窗长和窗宽
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        # 把21个点画出来
        if result.multi_hand_landmarks:
            # 对每个点进行循环
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLms_Style, handCon_Style)
                # 得到21个点的坐标，得到得是比例
                # 循环每只手的每个点（i从0到20）
                for i, lm, in enumerate(handLms.landmark):
                    # 根据前面得到的窗长和窗宽得到真实的x和y的位置，并转换成整数
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos + 25, yPos - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (0, 0, 255), 2)
                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 20, (166, 56, 56), cv2.FILLED)
                    print(i, xPos, yPos)
        # 计算得到每秒传输帧数
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 3)

        # 把每一帧显示出来
        cv2.imshow('img', img)
    # 卡1ms跳出
    if cv2.waitKey(0) == ord('q'):
        break