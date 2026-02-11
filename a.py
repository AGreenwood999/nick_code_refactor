import cv2

data = ["data/U2026Ca_3379.wmv", "data/U2026Ca_3379_normalized.wmv"]

images = [cv2.VideoCapture(d) for d in data]


while True:
    a = [i.read() for i in images]
    print(a[0][1])

    cv2.imshow("", a[0][1])
    cv2.waitKey(0)
