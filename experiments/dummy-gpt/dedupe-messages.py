import re

# Keep only lowercase Latin/Cyrillic letters, digits, spaces and common punctuation.
# Everything else is dropped: emoji, zero-width chars (e.g. \u034f), em/en dashes,
# box-drawing and other rare symbols that would just add noise to the tokenizer.
KEEP = re.compile(r"[^a-zа-яё0-9 .,!?:;'\"()\-+=*/@#%&\[\]<>_^]")

# Twitch usernames (Latin letters, digits, underscore) after an @. Replace the whole
# mention with a constant token so the model learns "a mention goes here" without
# memorizing real handles or bloating the vocabulary with one-off Latin strings.
MENTION = re.compile(r"@[a-z0-9_]+")


def clean(line: str) -> str:
    line = line.strip().lower()
    if "http" in line:  # drop messages that are mostly long Discord URLs
        return ""
    line = MENTION.sub("@user", line)
    return KEEP.sub("", line)


with open("messages.txt", encoding="utf-8") as f:
    lines = f.readlines()

seen = set()
unique_lines = []
for line in lines:
    line_clean = clean(line)
    if line_clean and line_clean not in seen:
        seen.add(line_clean)
        unique_lines.append(line_clean + "\n")

with open("messages_deduped.txt", "w", encoding="utf-8") as f:
    f.writelines(unique_lines)

print(f"Source length: {len(lines)}; Output length: {len(unique_lines)}")
