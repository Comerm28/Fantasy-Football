import re
import csv

with open('data/pdf_copy_of_espn_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'(\d+)\.\s+\(([^)]+)\)\s+([A-Za-z\'\. ]+),\s+([A-Z]+)\s+\$\d+\s+(\d+)'

matches = re.findall(pattern, text)

with open('data/espn_top_300.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Overall Rank', 'Position Rank', 'Name', 'Team', 'Bye Week'])  # headers
    for match in matches:
        writer.writerow(match)
