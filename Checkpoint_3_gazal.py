#!/usr/bin/env python
# coding: utf-8

import time
import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
#import org.openqa.selenium.support.ui.Select;
#import selenium.webdriver.support.select
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.service import Service

op = webdriver.ChromeOptions()
op.add_argument('--no-sandbox')
op.add_argument('--verbose')
op.add_argument("--disable-notifications")
op.add_experimental_option("prefs", {
  "download.default_directory": "C:\\Users\\gazal\\Downloads\\",
  "download.prompt_for_download": False,
  "download.directory_upgrade": True,
  "safebrowsing.enabled": True})
op.add_argument('--disable-gpu')
op.add_argument('--disable-software-rasterizer')
ser = Service(executable_path='./chromedriver.exe')
opt=webdriver.ChromeOptions()
# prefs={"download.default_directory":r"C:\Users\gazal\Downloads"}
#options.add_experimental_option("prefs",prefs)

driver = webdriver.Chrome(service=ser, options=opt)

#directing to the website to fetch data
#driver.get("https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2023/pvqr-7yc4")
driver.get("https://data.cityofnewyork.us/City-Government/Open-Parking-and-Camera-Violations/nc67-uf89")

#clicking on view data
driver.find_element(By.LINK_TEXT,'View Data').click()
time.sleep(4)

#clicking on issue date filtering menu
driver.find_element(By.XPATH,"//*[@id='column-header-issue_date-4']/div[1]/button").click()
time.sleep(4)

#clicking on filter button from dropdown
driver.find_element(By.ID,"column-header-issue_date-4-dropdown-picklist-0").click()
time.sleep(4)

#########################################################################################

#selecting desired start date and time from calender

#on the filter pane clicking on the start date input box
driver.find_element(By.XPATH,"//*[@id='ast-node-47-13']/div/div[1]/div/input").click()
time.sleep(4)

#selecting start month year
month_year_format = "%B %Y"
desired_month_year = "January 2021"
desired_month_year = datetime.strptime(desired_month_year,month_year_format)

while True:
    current_month_year_display = driver.find_element(By.XPATH, "//*[@id='ast-node-47-13']/div/div[2]/div[2]/div/div/div[2]/div[1]/div[1]").text
    current_month_year_display = datetime.strptime(current_month_year_display,month_year_format) 
    #time.sleep(4)
    
    if current_month_year_display == desired_month_year:
        break
        
    elif(current_month_year_display > desired_month_year):
        driver.find_element(By.XPATH,"//*[@id='ast-node-47-13']/div/div[2]/div[2]/div/div/button[1]").click()
        #time.sleep(4)
    
    else:
        driver.find_element(By.XPATH," //*[@id='ast-node-47-13']/div/div[2]/div[2]/div/div/button[2]").click()
        #time.sleep(4)

#selecting desired date
desired_date = 1
dates = driver.find_elements(By.XPATH,"//*[contains(@class,'react-datepicker__day react-datepicker__day--')]")
#print(dates)

#fetching the date text for all the dates present in the month
date_text = []
i=0
for every_date in dates:
    date_element_text = int(dates[i].text)
    i=i+1
    date_text.append(date_element_text)
#print(date_text)


#selecting the index of the date_text to particulary fetch anc click on that element
i=0
for date in date_text:
    if date==desired_date:
        date_index = i
        #print(date_index)
        dates[i].click()
        break
    else:
        i=i+1
      
time.sleep(4)
time.sleep(4)

#on the filter pane clicking on the start date input box again to select the time
driver.find_element(By.XPATH,"//*[@id='ast-node-47-13']/div/div[1]/div/input").click()
time.sleep(4)

#selecting the time
driver.find_element(By.XPATH,"//*[@id='ast-node-47-13']/div/div[2]/div[2]/div/div/div[3]/div[2]/div/ul/li[1]").click()
time.sleep(4)

########################################################################################

#selecting desired end date and time from calender

# getting today's date
today = datetime.today().date()

#on the filter pane clicking on the end date input box
driver.find_element(By.XPATH,"//*[@id='ast-node-48-9']/div/div[1]/div/input").click()
time.sleep(4)

today_month_year = today.strftime("%B %Y")
today_month_year = datetime.strptime(today_month_year, month_year_format)

while True:
    current_month_year_display = driver.find_element(By.XPATH, "//*[@id='ast-node-48-9']/div/div[2]/div[2]/div/div/div[2]/div[1]/div[1]").text
    current_month_year_display = datetime.strptime(current_month_year_display,month_year_format) 
    #time.sleep(4)
    
    if current_month_year_display == today_month_year:
        break
        
    elif(current_month_year_display > today_month_year):
        driver.find_element(By.XPATH,"//*[@id='ast-node-48-9']/div/div[2]/div[2]/div/div/button[1]").click()
        #time.sleep(4)
    
    else:
        driver.find_element(By.XPATH,"//*[@id='ast-node-48-9']/div/div[2]/div[2]/div/div/button[2]").click()
        #time.sleep(4)
        
#selecting desired date
today_date = int(today.strftime("%d"))
dates = driver.find_elements(By.XPATH,"//*[contains(@class,'react-datepicker__day react-datepicker__day--')]")
#print(dates)

#fetching the date text for all the dates present in the month
date_text = []
i=0
for every_date in dates:
    date_element_text = int(dates[i].text)
    i=i+1
    date_text.append(date_element_text)
#print(date_text)

#selecting the index of the date_text to particulary fetch anc click on that element
i=0
for date in date_text:
    if date==today_date:
        date_index = i
        #print(date_index)
        dates[i].click()
        break
        
    else:
        i=i+1

        
time.sleep(4)

#on the filter pane clicking on the end date input box again to select the time
driver.find_element(By.XPATH,"//*[@id='ast-node-48-9']/div/div[1]/div/input").click()
time.sleep(4)

#selecting the time
driver.find_element(By.XPATH,"//*[@id='ast-node-48-9']/div/div[2]/div[2]/div/div/div[3]/div[2]/div/ul/li[96]").click()
time.sleep(4)

##############################################################################

#applyig the filter criteria by clicking on the apply button
driver.find_element(By.ID,"apply-button").click()

#clicking on issue date filtering menu
#driver.find_element(By.XPATH,"//*[@id='column-header-issue_date-4']/div[1]/button").click()
#time.sleep(4)

#sorting the data in ascending order
#driver.find_element(By.ID,"column-header-issue_date-4-dropdown-picklist-1").click()
time.sleep(4)

#clicking on the export button
driver.find_element(By.XPATH,"//*[@id='grid-ribbon-export-button']/forge-button/button").click()
time.sleep(4)



# to download file
#driver.find_element(By.XPATH,"//*[@id='export-button-group']/forge-button-toggle[1]").click()
#time.sleep(4)

#click on the dropdown button
#selectElement = driver.find_element(By.XPATH,"/html/body/forge-dialog/div/forge-scaffold/div[2]/div[3]/forge-select//div")

#selectElement.select_by_visible_text("CSV for Excel")
#time.sleep(4)

# click on the download button
driver.find_element(By.XPATH,"/html/body/forge-dialog/div/forge-scaffold/div[3]/forge-toolbar/forge-button[2]/button").click()
   
# to download csv file
#driver.find_element(By.ID,"//*[@id='grid-ribbon-export-button']/forge-button/button/span").click()
#time.sleep(4)

#or to download csv for excel file
#driver.find_element(By.ID,"//*[@id='grid-ribbon-export-button']/forge-button/button/span").click()
#time.sleep(4)


# to download api endpoint


#selectt = Select()
#driver.get(" ")
#driver.find_element(By.LINK_TEXT,'View Data').send_keys('dp@gmail.com')
#filter_button.click()
#drop.select_by_visible_text("Filter")
#until(EC.element_to_be_clickable((issue_date_menu))).click()
#wait = WebDriverWait(driver, 50)
#filter_button=wait.until(EC.element_to_be_clickable((By.ID,"column-header-issue_date-4-dropdown-picklist-0")))
#driver.Select(find_element(By.ID,"column-header-issue_date-4")).click()
#driver.find_element(By.XPATH,"//button[normalize-space()='Export']//span[@class='forge-button__ripple']").click()
#driver.find_element(By.XPATH,"//html//body//forge-dialog//div//forge-scaffold//div[3]//forge-toolbar//forge-button[2]//button//span").click()
#time.sleep(4)
