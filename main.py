import re
import time
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import cv2
import numpy as np

from model import Model  # 你的训练模型模块，确保同目录或正确路径


class BilibiliLogin:
    def __init__(self, username, password):
        self.url = 'https://passport.bilibili.com/login'

        # 若用以下方式运行要挂梯子，网络可能会影响效果
        self.options = webdriver.ChromeOptions()
        self.browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

        self.browser.maximize_window()
        self.wait = WebDriverWait(self.browser, 30)  # 等待时间延长到30秒
        self.username = username
        self.password = password
        self.model = Model()  # 初始化你的模型

    def open(self):
        self.browser.get(self.url)
        user_input = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="请输入账号"]'))
        )
        password_input = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="请输入密码"]'))
        )
        user_input.send_keys(self.username)
        password_input.send_keys(self.password)

        login_btn = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.btn_primary'))
        )
        time.sleep(1)
        login_btn.click()

    def pick_code(self, round_num):
        # 等待验证码弹出
        time.sleep(3)

        # 定位验证码背景div和内部显示的img元素
        wrap_div = self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'geetest_item_wrap'))
        )
        img_ele = wrap_div.find_element(By.CLASS_NAME, 'geetest_item_img')

        # 从div的style属性里提取验证码背景图链接
        style = wrap_div.get_attribute('style')
        match = re.search(r'url\("(.*?)"\)', style)
        if not match:
            raise Exception("未能提取验证码背景图链接")
        img_url = match.group(1)

        print(f"验证码图片地址: {img_url}")

        # 下载验证码图片
        img_content = requests.get(img_url).content

        # 保存验证码原图
        captcha_filename = f"raw_pic/bili_captcha_{round_num}.jpg"
        with open(captcha_filename, "wb") as f:
            f.write(img_content)
        print(f"已保存验证码原图到 raw_pic/{captcha_filename}")

        # 用你的模型识别验证码
        small_img, big_img = self.model.detect(img_content)
        print(f"检测到小图: {len(small_img.keys())}个, 大图: {len(big_img)}个")

        order_imgs = self.model.split_order_image(count=len(small_img.keys()))
        result_list = self.model.siamese_from_order(order_imgs, big_img)
        print("模型识别点击坐标：", result_list)

        # 标注识别结果并保存图片
        img_array = np.frombuffer(img_content, np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        for i, (x, y) in enumerate(result_list):
            draw_x = x + 35  # 偏移量和模型识别时保持一致
            draw_y = y + 35
            cv2.circle(original_img, (draw_x, draw_y), 10, (0, 0, 255), 2)
            cv2.putText(original_img, str(i + 1), (draw_x, draw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # 保存标注图
        marked_filename = f"marked_pic/bili_result_{round_num}.jpg"
        cv2.imwrite(marked_filename, original_img)
        print(f"已保存标注图到 {marked_filename}")


        for i, (x, y) in enumerate(result_list):
            # 原图中标注位置 + 偏移
            offset_x = x + 30
            offset_y = y + 30

            print(f"准备点击第{i+1}个位置: offset=({offset_x:.1f}, {offset_y:.1f})")

            click_script = f"""
                var canvas = arguments[0];
                var rect = canvas.getBoundingClientRect();
                var x = rect.left + {offset_x};
                var y = rect.top + {offset_y};
                var clickEvent = new MouseEvent('click', {{
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: x,
                    clientY: y
                }});
                canvas.dispatchEvent(clickEvent);
            """
            self.browser.execute_script(click_script, img_ele)

            time.sleep(1)


        # 点击确认按钮
        confirm_btn = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.geetest_commit'))
        )
        confirm_btn.click()
        time.sleep(2)

    def close(self):
        self.browser.quit()


if __name__ == "__main__":
    USERNAME = "18223540359"
    PASSWORD = "20040907"

    for i in range(10):
        print(f"\n================= 第 {i + 1} 次验证码识别测试 =================")

        bilibili = BilibiliLogin(USERNAME, PASSWORD)
        try:
            bilibili.open()
            bilibili.pick_code(i + 1)  # 传递当前轮次
        except Exception as e:
            print(f"第 {i + 1} 次出错了:", e)
            bilibili.browser.save_screenshot(f"debug/error_debug_{i + 1}.png")
        finally:
            bilibili.close()
            time.sleep(1)


