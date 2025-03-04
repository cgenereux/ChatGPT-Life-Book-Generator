# ChatGPT-Life-Book-Generator

# Abstract

This tool allows you to turn many hundreds or thousands of past ChatGPT conversations into an in-depth, and personal book in bullet points for AIs to know and help you better! Think about it like giving ChatGPT substatially more memory or an infinite context window. Using the recommended 500 number of conversations it should cost about $3.00 in API credits and take about 18 minutes.

Info
Setup Time: ~5-10 minutes
Completely Open Source: I charge nothing for this
Supports Large Exports: Handles thousands of conversations efficiently

# Steps

# 1. Exporting your ChatGPT data 

(If you have 8000+ conversations like me, it might take a bit longer to export or require a 2nd try)

Log in to ChatGPT at chat.openai.com
Click your profile in the top-right corner
Select "Settings"
Navigate to "Data controls"
Click "Export data"
Wait for the export to be prepared and download it
Unzip the file and locate the conversations.json file

# 2. Installation.

Prerequisites
Python 3.6 or higher
An Anthropic API key for Claude (this just takes about 2 minutes to setup but the minimum amount of credits you can buy is $5: https://console.anthropic.com/settings/limits)
Your ChatGPT conversation export (in JSON format)
Steps

Clone or download this repository
Install required dependencies:
Copypip install anthropic

Place your ChatGPT conversation export file (named conversations.json) in the same directory as the script

