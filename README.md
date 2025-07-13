# **ChatGPT Life Book Generator**

## **Abstract**

Turn hundreds of past ChatGPT conversations into a detailed, bullet-point book to unlock a more powerful memory for ChatGPT.

## **How to Use It**

### **Step 1: Extract your ChatGPT Data**

(If you have 8000+ conversations like me, it might take longer to export or require a second try)

- Click your profile icon in the top right corner
- Go to **Settings → Data controls → Export data**
- Wait for your data export and download it
- Unzip the downloaded file and grab the **`conversations.json`**

### **Step 2: installation**

#### What you’ll need:

- Python 3.6 or newer
- An Anthropic API key with some funds (create one: [here](https://console.anthropic.com/settings/keys))

#### Quick setup:

- Clone or download the project 
- Open your editor or command line and run:

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
- How many recent conversations you'd like to use (I think the sweet spot is 300-500)

## **Final notes**

You'll find the finished Context Book as a markdown file (`.md`) in the **Books** folder but converting it to a PDF can be convenient.

I hope you find it useful!
