import re
import csv

# Read the txt file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Regex pattern to match player entries
pattern = r'(\d+)\.\s+\(([^)]+)\)\s+([A-Za-z\'\. ]+),\s+([A-Z]+)\s+\$\d+\s+(\d+)'

# Find all matches
matches = re.findall(pattern, text)

# Prepare CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Overall Rank', 'Position Rank', 'Name', 'Team', 'Bye Week'])  # headers
    for match in matches:
        writer.writerow(match)
