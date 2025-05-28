import base64
import random
from pathlib import Path
from typing import List, Any
import ddddocr
import cv2
import time
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

bottom = 0.87  # 底部区域裁剪比例
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_CHANNEL_DIR = Path("channel")
RAW_IMAGE_DIR = Path("./raw_pic")
BASE64_PREFIX = "data:image/png;base64,"
MIN_BOX_SIZE = 10
ASPECT_RATIO_LIMIT = 8
RANDOM_SLEEP_RANGE = (2, 5)

class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def save_v_channel(image_path: Path, output_dir: Path = DEFAULT_CHANNEL_DIR) -> Path:
        """保存图像的明度通道"""
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        output_path = output_dir / f"v_channel_{image_path.name}"
        cv2.imwrite(str(output_path), hsv[:, :, 2])
        print(f"已保存明度通道到：{output_path}")
        return output_path

    @staticmethod
    def process_image(image_path: Path, detector: ddddocr.DdddOcr, recognizer: ddddocr.DdddOcr, img_ele) -> List[
        tuple[str, tuple]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        height, width = img.shape[:2]
        cropped_height = int(height * 0.85)
        img_top_85 = img[:cropped_height, :]

        # 将裁剪图像编码成 PNG 字节传给 detector
        _, buffer = cv2.imencode('.png', img_top_85)
        cropped_image_bytes = buffer.tobytes()

        # 获取合并后的检测框（相对于上85%图像）
        bboxes = detect_and_merge(detector, cropped_image_bytes)

        bboxes.sort(key=lambda box: box[0])
        results = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # ✅ y1, y2 是相对于裁剪图像的，需要修正为原图坐标
            if y2 > cropped_height:
                continue  # ✅ 过滤掉越界框（万一误识别）

            w, h = x2 - x1, y2 - y1
            if any([w < MIN_BOX_SIZE, h < MIN_BOX_SIZE, max(w / h, h / w) > ASPECT_RATIO_LIMIT]):
                continue

            # 在原图中裁剪（注意 img 用的是原图）
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            text = recognizer.classification(cv2.imencode('.png', cropped)[1].tobytes())
            if text.strip():
                results.append((text, (x1, y1, x2, y2)))
                ImageProcessor._draw_result(img, x1, y1, x2, y2, text)

        result_path = DEFAULT_OUTPUT_DIR / f"result_{image_path.name}"
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(result_path), img)
        print(f"🎯 标注结果已保存至：{result_path}")
        return results

    @staticmethod
    def _draw_result(img: cv2.Mat, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


class WebCrawler:
    """网页爬取工具类"""

    def __init__(self):
        self.driver = self._init_driver()
        self.detector = ddddocr.DdddOcr(det=True)
        self.recognizer = ddddocr.DdddOcr()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @staticmethod
    def _init_driver() -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        try:
            service = Service(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=options)
        except Exception as e:
            print(f"❌ WebDriver 启动失败: {e}")
            raise

    def _process_image(self, prompt_text: str, img_ele) -> bool:
        img_path = RAW_IMAGE_DIR / f"{prompt_text}.png"
        try:
            channel_path = ImageProcessor.save_v_channel(img_path)
            results = ImageProcessor.process_image(channel_path, self.detector, self.recognizer, img_ele)
            print("🧠 识别结果:", results)
            return True
        except Exception as e:
            print(f"❌ 图像处理失败: {str(e)}")
            return False

    def _random_delay(self) -> None:
        time.sleep(random.uniform(*RANDOM_SLEEP_RANGE))
        self.driver.refresh()

    def _simulate_clicks(self, img_ele, image_path: Path, click_sequence: List[tuple[str, tuple[int, int]]]):
        """
        在网页 canvas 上模拟点击 click_sequence 中的每个坐标。

        :param img_ele: 网页中的 canvas 或 img 元素（Selenium WebElement）
        :param image_path: 当前处理图像的路径（用于日志打印）
        :param click_sequence: 格式为 [(text, (x, y)), ...]
        """
        print(f"🖱️ 开始模拟点击 {image_path.name} 的目标点，共 {len(click_sequence)} 个")

        for i, (text, (x, y)) in enumerate(click_sequence):
            # 偏移坐标（可根据实际 canvas 显示情况调整）
            offset_x = x + 30
            offset_y = y + 30

            print(f"🎯 第{i + 1}个目标: 文字='{text}', 原始坐标=({x}, {y})，偏移后=({offset_x}, {offset_y})")

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

            try:
                self.browser.execute_script(click_script, img_ele)
                time.sleep(1)  # 防止太快被判定为机器人
            except Exception as e:
                print(f"❌ 第{i + 1}个点击失败：{e}")

        print(f"✅ 点击完成：{image_path.name}")


    def refresh_page(self):
        """刷新页面"""
        self.driver.refresh()


def fill_click_sequence(results, prompt, v_channel_path, detector, myocr, img_ele):
    click_sequence = []
    used_bboxes = set()

    for char in prompt:
        found = next(((t, b) for t, b in results if t == char and str(b) not in used_bboxes), None)
        if found:
            used_bboxes.add(str(found[1]))
        click_sequence.append(found or (char, None))

    print("📌 初步匹配:", click_sequence)

    # 补充自定义模型识别（可选，如果你有 myocr 模型）
    if any(b is None for _, b in click_sequence):
        alt_results = ImageProcessor.process_image(v_channel_path, detector, myocr, img_ele)
        for i, (char, bbox) in enumerate(click_sequence):
            if bbox is None:
                found = next(((t, b) for t, b in alt_results if t == char and str(b) not in used_bboxes), None)
                if found:
                    click_sequence[i] = found
                    used_bboxes.add(str(found[1]))

    print("🔁 补充后:", click_sequence)

    # 随机填充空白项（模拟点击一个合理但非目标的位置）
    remaining_bboxes = [b for _, b in results if str(b) not in used_bboxes]
    random.shuffle(remaining_bboxes)

    for i, (char, bbox) in enumerate(click_sequence):
        if bbox is None and remaining_bboxes:
            click_sequence[i] = (char, remaining_bboxes.pop())

    print("✅ 最终点击序列:", click_sequence)
    return click_sequence


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

    def get_pic(self, round_num):
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
        return Path(captcha_filename), img_ele

    def check_success(self, timeout=5):
        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                text = self.browser.execute_script("""
                    var el = document.querySelector('div.geetest_result_tip.geetest_success');
                    return el ? el.innerText.trim() : '';
                """)
                if text:
                    print(f"[调试] JS 获取成功提示文本: '{text}'")
                    return text.startswith("验证成功")
                time.sleep(0.2)
            print("[调试] 超时未通过验证")
            return False
        except Exception as e:
            print(f"[调试] check_success 异常: {e}")
            return False


    def close(self):
        self.browser.quit()

def merge_close_bboxes(bboxes, distance_threshold=10):
    """
    合并彼此靠近的检测框。

    :param bboxes: 原始检测框列表 [(x1, y1, x2, y2), ...]
    :param distance_threshold: 合并判定的最大距离（像素）
    :return: 合并后的框列表
    """
    merged = []
    used = [False] * len(bboxes)

    for i, box_i in enumerate(bboxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = box_i
        merged_box = [x1, y1, x2, y2]
        used[i] = True
        for j, box_j in enumerate(bboxes):
            if used[j]:
                continue
            a = merged_box
            b = box_j
            h_dis = max(0, max(b[0] - a[2], a[0] - b[2]))  # 水平距离
            v_dis = max(0, max(b[1] - a[3], a[1] - b[3]))  # 垂直距离
            if max(h_dis, v_dis) < distance_threshold:
                merged_box[0] = min(a[0], b[0])
                merged_box[1] = min(a[1], b[1])
                merged_box[2] = max(a[2], b[2])
                merged_box[3] = max(a[3], b[3])
                used[j] = True
        merged.append(tuple(merged_box))
    return merged


def detect_and_merge(detector, img_bytes) -> List[tuple[int, int, int, int]]:
    """
    检测图像中的目标框并合并。

    :param detector: ddddocr 的 detector 实例
    :param img_bytes: 输入图像的字节数据（PNG编码）
    :return: 合并后的框列表
    """
    bboxes = detector.detection(img_bytes)
    print("原始框数量:", len(bboxes))

    merged_boxes = merge_close_bboxes(bboxes)
    print("合并后框数量:", len(merged_boxes))

    return merged_boxes



def recognize_bottom_text(img_path):
    """
    裁剪图像底部区域并进行 OCR 识别。

    :param img_path: 图像路径
    :return: 识别结果文本
    """
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    crop_img = image[int(h * bottom):, :]

    _, buffer = cv2.imencode(".jpg", crop_img)
    img_bytes = buffer.tobytes()

    ocr = ddddocr.DdddOcr()
    result = ocr.classification(img_bytes)
    print("底部区域识别结果：", result)
    return result




# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    USERNAME = "18223540359"
    PASSWORD = "11111111"
    total = 10
    success_count = 0
    detector = ddddocr.DdddOcr(det=True)
    recognizer = ddddocr.DdddOcr()
    myocr = ddddocr.DdddOcr(det=True,
                            import_onnx_path="models/bili_captcha_0.4074074074074074_250_7000_2025-05-29-00-42-07.onnx",
                            charsets_path="models/charsets.json")

    with WebCrawler() as crawler:

        for i in range(total):
            print(f"\n================= 第 {i + 1} 次验证码识别测试 =================")

            bilibili = BilibiliLogin(USERNAME, PASSWORD)
            try:
                bilibili.open()
                img_path, img_ele = bilibili.get_pic(i)

                prompt = recognize_bottom_text(img_path)

                # 第一步：提取明度图并识别字符
                v_channel_path = ImageProcessor.save_v_channel(img_path)
                results = ImageProcessor.process_image(v_channel_path, detector, recognizer, img_ele)

                # 第二步：根据 prompt 构造点击序列
                click_sequence = fill_click_sequence(results, prompt)

                # 第三步：模拟点击
                crawler._simulate_clicks(img_ele, img_path, click_sequence)

                if bilibili.check_success():  # 检查是否识别成功
                    print("✅ 第 {} 次验证成功".format(i + 1))
                    success_count += 1
                else:
                    print("❌ 第 {} 次验证失败".format(i + 1))
            except Exception as e:
                print(f"🚫 第 {i + 1} 次出错了:", e)
                bilibili.browser.save_screenshot(f"debug/error_debug_{i + 1}.png")
            finally:
                bilibili.close()
                time.sleep(1)

    print(f"\n识别成功次数: {success_count}/{total}")
    print(f"识别成功率: {success_count / total * 100:.2f}%")

