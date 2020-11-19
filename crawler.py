
# -*- coding: utf-8-sig -*-


import os
import copy
import sys
import numpy as np
import pandas as pd



from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui


from selenium import webdriver
from selenium.webdriver.common.keys import Keys







# ----------------------------------------------------------------------------------
# Main Window
class ReviewTraining(QMainWindow):
    def __init__(self):
        super().__init__()
        #
        # -------------------------------------------------------------------
        # 윈도우 특성 설정
        self.setWindowTitle('ReviewCrawler')    # 윈도우 타이클 지정
        self.setGeometry(0, 0, 600, 300)       # 윈도우 위치/크기 설정
        #
        # 탭 설정
        self.tabs = QTabWidget(self)
        self.tabs.setGeometry(QtCore.QRect(0, 0, 600, 300))
        #
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Crawling")
        #
        # ======================================================================================
        # Tab1
        self.label_mall = QLabel('쇼핑몰선택', self.tab1)
        self.label_mall.setGeometry(QtCore.QRect(10, 30, 100, 30))
        self.label_mall.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        #
        self.combo_mall = QComboBox(self.tab1)
        self.combo_mall.setGeometry(QtCore.QRect(120, 30, 150, 30))
        self.combo_mall.addItems(['쿠팡'])        # "Coupang": "https://www.coupang.com"
        #
        self.btn_open_mall = QPushButton('페이지열기', self.tab1)
        self.btn_open_mall.setGeometry(QtCore.QRect(280, 30, 100, 30))
        self.btn_open_mall.clicked.connect(self.open_mall)
        #
        self.label_notice = QLabel('Review data를 수집하려는 제품을 수동으로 선택하신 후 하단의 [리뷰데이터 수집하기] '
                                   '버튼을 클릭해 주세요')
        self.label_notice.setGeometry(QtCore.QRect(10, 70, 590, 60))
        self.label_notice.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        #
        self.label_reviews_file = QLabel('리뷰 데이터파일: reviews_crawling.xlsx', self.tab1)
        self.label_reviews_file.setGeometry(QtCore.QRect(10, 200, 290, 30))
        self.label_reviews_file.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        #
        self.btn_save_review = QPushButton('리뷰데이터 수집하기', self.tab1)
        self.btn_save_review.setGeometry(QtCore.QRect(300, 200, 150, 30))
        self.btn_save_review.clicked.connect(self.save_reviews)
        #
        self.label_result = QLabel('', self.tab1)
        self.label_result.setGeometry(QtCore.QRect(10, 340, 590, 30))
        self.label_result.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        #
        # ======================================================================================
        self.pd_reviews = pd.DataFrame()
        self.show()
    #
    # =======================================================================================================
    # Tab1: Open Shopping Mall
    def open_mall(self):
        key = self.combo_mall.currentText()
        url = 'https://www.coupang.com'
        #
        self.driver = webdriver.Chrome('chromedriver/chromedriver.exe')
        self.driver.implicitly_wait(2)  # seconds
        self.driver.get(url)
    #
    # -----------------------------------------------------------------------------------------------------
    # Tab1: Crawling Review data
    def save_reviews(self):
        review_file = 'dataset/reviews_crawling.xlsx'
        print('review crawling')
        try:
            self.label_result.setText('데이터 수집중')
            try:
                self.pd_reviews = pd.read_excel(review_file, encoding='utf-8-sig')
                print(len(self.pd_reviews))
            except:
                self.pd_revies = {}
            #
            tabs = self.driver.window_handles
            self.driver.switch_to.window(window_name=tabs[-1])
            url = self.driver.current_url
            new_reviews = []
            #
            # 쿠팡
            prod_category = self.driver.find_element_by_id('breadcrumb').text.replace('\n', ' ')
            prod_name = self.driver.find_element_by_class_name('prod-buy-header').find_element_by_tag_name('h2').text
            jjim_val = ''
            # ---------------------------------------------------------------------------
            # click 상품평
            self.driver.find_element_by_class_name('product-tab-review-count').click()
            # ---------------------------------------------------------------------------
            descs = self.driver.find_elements_by_class_name('js_reviewArticleContent')
            for desc in descs:
                new_reviews.append([url, prod_category, prod_name, jjim_val, desc.text])
            #
            pd_new_reviews = pd.DataFrame(new_reviews)
            pd_new_reviews.columns = ['url', 'category', 'product', 'jjim_cnt', 'review']
            #
            self.pd_reviews = self.pd_reviews.append(pd_new_reviews)
            self.pd_reviews.to_excel(review_file, index=False, header=True, encoding='utf-8-sig')
            self.label_result.setText('리뷰데이터가 정상적으로 저장되였습니다')
        except Exception as ex:
            print(str(ex))




# =======================================================================================================
def main():
    app = QApplication(sys.argv)
    win = ReviewTraining()
    sys.exit(app.exec_())




# =======================================================================================================
if __name__ == '__main__':
    main()



