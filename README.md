# **ChatGPT Life Book Generator**

## **Abstract**

This tool turns hundreds  of your past ChatGPT conversations into a detailed, personalized bullet-point book. This lets you give ChatGPT a significantly more powerful memory that let's it understand and help you better. 500 conversations typically cost about $3 in API credits and takes roughly 18 minutes.

## **Quick Info**

- ⚡ **Fast Setup:** Takes 5-10 minutes to set up
- 📂 **Fully Open Source:** No monetization 
- 📦 **Handles Big Exports**: Can process thousands of conversations

## **How to Use It**

### **Step 1: Extract your ChatGPT Data**

(If you have 8000+ conversations like me, it might take longer to export or require a second try)

- Click your profile icon in the top right corner
- Go to **Settings → Data controls → Export data**
- Wait for your data export and download it
- Unzip the downloaded file and grab the **`conversations.json`**

### **Step 2: Install the Stuff**

#### What you’ll need:

- Python 3.6 or newer
- An Anthropic API key with some funds (create one: [here](https://console.anthropic.com/settings/keys))

#### Quick setup:

- Clone or download the repo
- Open a terminal/command line and run:

```bash
pip install anthropic
```

- Move your **`conversations.json`** file into the folder with the script

### **Step 3: Generate your Book**

In your terminal, run:

```bash
python generate_book.py
```

When asked, enter:

- Your Anthropic API key
- How many recent convos you want to use

## **Where’s the Book?**

You'll find the finished Context Book as a markdown file (`.md`) in the **Books** folder but converting it to a PDF can be convenient.

I hope you all find it useful!
