import json
import time
from model import Model
import httpx
import cv2
import numpy as np

offset_x = 20
offset_y = 20

t = time.time()

model = Model()

for retry in range(1):
    tt = time.time()
    with open("35126e7a1779425f880154ec81e1e0ba.jpg", "rb") as f:
        pic_content = f.read()
    print(time.time() - tt)

    # 读取原图（用于标注）
    img_array = np.frombuffer(pic_content, np.uint8)
    original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    ttt = tt = time.time()
    small_img, big_img = model.detect(pic_content)
    print(
        f"检测到小图: {len(small_img.keys())}个,大图: {len(big_img)} 个,用时: {time.time() - tt}s"
    )
    tt = time.time()
    order_imgs = model.split_order_image(count=len(small_img.keys()))
    result_list = model.siamese_from_order(order_imgs, big_img)
    print(f"文字配对完成,用时: {time.time() - tt}")
    point_list = []
    print("结果点坐标：", result_list)

    for i, (x, y) in enumerate(result_list):
        draw_x = x + 15 + offset_x
        draw_y = y + 15 + offset_y
        cv2.circle(original_img, (draw_x, draw_y), 10, (0, 0, 255), 2)  # 红色圆圈
        cv2.putText(original_img, str(i + 1), (draw_x, draw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存结果图像
    cv2.imwrite("marked_pic/marked_result.jpg", original_img)
    print("已将匹配位置标注到 marked_result.jpg")

    for i in result_list:
        left = str(round((i[0] + 30) / 333 * 10000))
        top = str(round((i[1] + 30) / 333 * 10000))
        point_list.append(f"{left}_{top}")
    wait_time = 2.0 - (time.time() - ttt)
    time.sleep(wait_time)
    tt = time.time()
total_time = time.time() - t
print(f"总计耗时(含等待{wait_time}s): {total_time}")
