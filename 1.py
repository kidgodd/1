import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """加载CSV文件并返回DataFrame"""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """标准化数据处理"""
    # 选择特征列
    features = [
        'A_follower_count', 'A_following_count', 'A_listed_count',
        'A_mentions_received', 'A_retweets_received', 'A_mentions_sent',
        'A_retweets_sent', 'B_follower_count', 'B_following_count',
        'B_listed_count', 'B_mentions_received', 'B_retweets_received',
        'B_mentions_sent', 'B_retweets_sent'
    ]

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled, data['Choice']


from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data_scaled, target):
    """训练模型并评估"""
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return model, report



import requests
from bs4 import BeautifulSoup


def fetch_weibo_data(username):
    url = f'https://weibo.com/{username}'  # 假设爬取微博个人页面
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # 发送请求
        response = requests.get(url, headers=headers)

        # 检查请求是否成功
        if response.status_code != 200:
            raise Exception(f"请求失败，状态码：{response.status_code}")

        # 解析网页
        soup = BeautifulSoup(response.text, 'html.parser')


        # 假设我们要提取关注者数量
        followers = soup.find('div', {'class': 'followers-class'})  # 这里需要根据实际页面结构调整
        if followers is not None:
            followers_count = followers.text.strip()
        else:
            followers_count = '未找到关注者信息'

        # 假设我们要提取发布的帖子数量
        posts = soup.find('div', {'class': 'posts-class'})  # 需要根据实际页面结构调整
        if posts is not None:
            posts_count = posts.text.strip()
        else:
            posts_count = '未找到帖子信息'

        return {
            'followers': followers_count,
            'posts': posts_count
        }

    except Exception as e:
        # 捕获异常并打印
        print(f"爬取失败：{e}")
        return None


import sys
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLineEdit, QProgressBar
from data.data_processing import load_data, preprocess_data
from data.model import train_model
from data.web_scraping import fetch_weibo_data


class ScrapeThread(QThread):
    """爬取微博数据的线程"""
    result_signal = pyqtSignal(str)  # 用于传递结果到主界面
    progress_signal = pyqtSignal(int)  # 用于更新进度条

    def __init__(self, username):
        super().__init__()
        self.username = username

    def run(self):
        """在后台执行微博数据爬取操作"""
        try:
            self.progress_signal.emit(0)  # 初始化进度
            user_info = fetch_weibo_data(self.username)  # 爬取微博数据
            self.progress_signal.emit(50)  # 爬取数据完成

            result_text = f"微博用户：{self.username}\n关注者：{user_info.get('followers')}\n发布的帖子：{user_info.get('posts')}"
            self.progress_signal.emit(100)  # 完成

        except Exception as e:
            result_text = f"爬取失败：{str(e)}"
            self.progress_signal.emit(100)  # 完成

        self.result_signal.emit(result_text)  # 发出信号将结果传递给主线程



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('社交网络用户影响力分析与建模')
        self.setGeometry(100, 100, 800, 600)

        # 初始化界面组件
        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton('上传CSV文件')
        self.upload_btn.clicked.connect(self.upload_csv)

        self.analyze_btn = QPushButton('分析数据')
        self.analyze_btn.clicked.connect(self.analyze_data)

        self.weibo_username_input = QLineEdit(self)
        self.weibo_username_input.setPlaceholderText('输入微博用户名')

        self.weibo_scrape_btn = QPushButton('爬取微博数据')
        self.weibo_scrape_btn.clicked.connect(self.scrape_weibo_data)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)

        # 新增进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # 添加到布局
        self.layout.addWidget(self.upload_btn)
        self.layout.addWidget(self.analyze_btn)
        self.layout.addWidget(self.weibo_username_input)
        self.layout.addWidget(self.weibo_scrape_btn)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.result_area)

        # 设置中心窗口
        central_widget = QWidget(self)
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        self.data = None  # 存储数据

    def upload_csv(self):
        """上传CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, '选择CSV文件', '', 'CSV Files (*.csv)')
        if file_path:
            try:
                self.data = load_data(file_path)

                # 简单检查数据
                if self.data.empty:
                    raise ValueError("数据为空，请检查CSV文件。")

                self.result_area.setText(f"已加载数据: {file_path}")
            except Exception as e:
                self.result_area.setText(f"加载数据失败：{str(e)}")
                print(f"Error loading data: {str(e)}")  # 控制台输出错误信息

    def analyze_data(self):
        """分析数据并训练模型"""
        if self.data is None:
            self.result_area.setText("请先上传CSV文件。")
            return

        try:
            # 数据预处理
            data_scaled, target = preprocess_data(self.data)

            # 训练模型
            model, report = train_model(data_scaled, target)

            # 将英文报告翻译成中文并格式化
            report_in_chinese = self.translate_report(report)

            # 显示格式化后的报告
            self.result_area.setHtml(report_in_chinese)

        except Exception as e:
            self.result_area.setText(f"分析数据时出错：{str(e)}")
            print(f"Error during analysis: {str(e)}")  # 控制台输出错误信息

    def translate_report(self, report):
        """将报告中的英文字段翻译为中文，并格式化为符合需求的表格"""
        translation_map = {
            'accuracy': '准确度',
            'precision': '精确率',
            'recall': '召回率',
            'f1-score': 'F1分数',
            'support': '支持度',
            'macro avg': '宏观平均',
            'weighted avg': '加权平均'
        }

        # 替换报告中的字段
        for english, chinese in translation_map.items():
            report = report.replace(english, chinese)

        # 使用HTML进行格式化表格
        formatted_report = """
        <h2><b>模型性能报告</b></h2>
        <hr>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <thead>
                <tr>
                    <th><b>用户</b></th>  <!-- 表头是用户 -->
                    <th><b>精确率</b></th>
                    <th><b>召回率</b></th>
                    <th><b>F1分数</b></th>
                    <th><b>支持度</b></th>
                    <th><b>准确度</b></th>
                    <th><b>加权平均</b></th>
                    <th><b>宏观平均</b></th>  <!-- 添加宏观平均列 -->
                </tr>
            </thead>
            <tbody>
        """

        # 保存用户数据
        users_data = {}

        lines = report.split('\n')

        # 只处理需要的行（去掉宏观平均和加权平均的行）
        for line in lines:
            if line.strip() and not ("宏观平均" in line or "加权平均" in line):  # 跳过宏观平均和加权平均
                columns = line.strip().split()

                if len(columns) >= 6:
                    user = columns[0]  # 用户名称（例如：用户1、用户2）
                    precision = float(columns[1])  # 精确率
                    recall = float(columns[2])  # 召回率
                    f1_score = float(columns[3])  # F1分数
                    support = columns[4]  # 支持度
                    accuracy = float(columns[5])  # 准确度
                    weighted_avg = columns[6] if len(columns) > 6 else ''  # 加权平均

                    # 将每个用户的指标值保存到字典中
                    users_data[user] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'support': support,
                        'accuracy': accuracy,
                        'weighted_avg': weighted_avg
                    }

                    # 将每个用户的指标值填入表格
                    formatted_report += f"""
                    <tr>
                        <td>{user}</td>  <!-- 显示用户 -->
                        <td>{precision:.2f}</td>
                        <td>{recall:.2f}</td>
                        <td>{f1_score:.2f}</td>
                        <td>{support}</td>
                        <td>{accuracy:.2f}</td>
                        <td>{weighted_avg}</td>
                        <td>-</td>  <!-- 初始宏观平均列为空 -->
                    </tr>
                    """

        # 计算宏观平均（针对精确率、召回率、F1分数、准确度）
        if len(users_data) == 2:
            user1 = list(users_data.keys())[0]
            user2 = list(users_data.keys())[1]

            # 计算每个指标的宏观平均
            macro_avg_precision = (users_data[user1]['precision'] + users_data[user2]['precision']) / 2
            macro_avg_recall = (users_data[user1]['recall'] + users_data[user2]['recall']) / 2
            macro_avg_f1_score = (users_data[user1]['f1_score'] + users_data[user2]['f1_score']) / 2
            # 继续计算宏观平均
            macro_avg_accuracy = (users_data[user1]['accuracy'] + users_data[user2]['accuracy']) / 2

            # 在表格中添加宏观平均行
            formatted_report += f"""
                        <tr>
                            <td><b>宏观平均</b></td>
                            <td>{macro_avg_precision:.2f}</td>
                            <td>{macro_avg_recall:.2f}</td>
                            <td>{macro_avg_f1_score:.2f}</td>
                            <td>-</td>
                            <td>{macro_avg_accuracy:.2f}</td>
                            <td>-</td>
                            <td>-</td>
                        </tr>
                        """

            # 完成表格并返回
        formatted_report += "</tbody></table>"
        return formatted_report

    def scrape_weibo_data(self):
        """爬取微博数据"""
        username = self.weibo_username_input.text().strip()
        if not username:
            self.result_area.setText("请输入微博用户名。")
            return

        # 启动爬取微博数据的线程
        self.scrape_thread = ScrapeThread(username)
        self.scrape_thread.result_signal.connect(self.display_result)
        self.scrape_thread.progress_signal.connect(self.update_progress_bar)
        self.scrape_thread.start()

    def display_result(self, result_text):
        """在文本框中显示爬取结果"""
        self.result_area.setText(result_text)

    def update_progress_bar(self, progress_value):
        """更新进度条"""
        self.progress_bar.setValue(progress_value)

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())


import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
