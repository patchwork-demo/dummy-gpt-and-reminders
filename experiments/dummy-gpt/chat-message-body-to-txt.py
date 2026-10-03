import json

with open('chat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bodies = (
    comment['message']['body']
    for comment in data.get('comments', [])
    if 'message' in comment and 'body' in comment['message']
)

with open('messages.txt', 'w', encoding='utf-8') as f:
    for body in bodies:
        clean_body = body.replace('\r\n', ' ').replace('\n', ' ')
        f.write(clean_body + '\n')
