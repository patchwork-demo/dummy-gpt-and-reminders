with open('messages.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

seen = set()
unique_lines = []
for line in lines:
    line_clean = line.strip()
    if line_clean and line_clean not in seen:
        seen.add(line_clean)
        unique_lines.append(line)

with open('messages_deduped.txt', 'w', encoding='utf-8') as f:
    f.writelines(unique_lines)

print(f"Source length: {len(lines)}; Output length: {len(unique_lines)}")
